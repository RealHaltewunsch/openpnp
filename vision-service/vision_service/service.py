"""Service core: ties the encoder, heads and manifest together.

Holds the encoder session for the whole process, loads heads lazily,
warms them before first use, and produces task-specific responses.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import time

import numpy as np
from PIL import Image

from .encoder import Encoder, EncoderLoadError
from .heads import HeadLoadError, HeadManager, clamp01, clamp_angle, status_for
from .manifest import ManifestError, load_manifest

log = logging.getLogger(__name__)

TASKS = ("bottom-vision", "homing")


class InferenceError(RuntimeError):
    """Raised for request-level or inference-level failures."""

    def __init__(self, code, detail, status_code=500):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status_code = status_code


class VisionService:
    def __init__(self, models_dir, provider=None):
        self.models_dir = models_dir
        self.manifest_path = models_dir / "manifest.json"
        self.manifest_error = None
        self.manifest = None
        self.encoder = None
        self.head_manager = None
        self._load_manifest_and_encoder(provider)

    # ---- startup ------------------------------------------------------

    def _load_manifest_and_encoder(self, provider):
        try:
            self.manifest = load_manifest(self.manifest_path)
        except ManifestError as e:
            self.manifest_error = str(e)
            log.error("Manifest rejected: %s", e)
            return

        providers = [provider] if provider else None
        try:
            self.encoder = Encoder(self.manifest.encoder, providers=providers)
        except EncoderLoadError as e:
            self.manifest_error = str(e)
            log.error("Encoder unavailable: %s", e)
            return

        self.head_manager = HeadManager(self.manifest, self.encoder)

    @property
    def encoder_ok(self):
        return self.encoder is not None

    # ---- health --------------------------------------------------------

    def health(self):
        if self.head_manager is not None:
            heads_health = self.head_manager.health()
        elif self.manifest is not None:
            heads_health = {
                task: {
                    "version": spec.version,
                    "encoder_version": spec.encoder_version,
                    "status": "load_error",
                    "warmed": False,
                }
                for task, spec in self.manifest.heads.items()
            }
        else:
            heads_health = {}

        degraded = (
            self.manifest_error is not None
            or not self.encoder_ok
            or any(h["status"] != "ok" for h in heads_health.values())
        )
        encoder_info = None
        if self.encoder is not None:
            encoder_info = {
                "version": self.encoder.spec.version,
                "session_id": self.encoder.session_id,
                "provider": self.encoder.active_providers[0],
                "providers_available": list(self.encoder.active_providers),
            }

        import onnxruntime as ort

        return {
            "status": "degraded" if degraded else "ok",
            "service": "openpnp-vision",
            "version": "0.1.0",
            "contract_version": self.manifest.contract_version if self.manifest else None,
            "onnx_runtime_version": ort.__version__,
            "provider": encoder_info["provider"] if encoder_info else None,
            "providers_available": encoder_info["providers_available"] if encoder_info else [],
            "encoder_version": encoder_info["version"] if encoder_info else None,
            "encoder_session_id": encoder_info["session_id"] if encoder_info else None,
            "encoder_sessions_created": 1 if encoder_info else 0,
            "heads": heads_health,
            "manifest_error": self.manifest_error,
        }

    # ---- inference -----------------------------------------------------

    def infer(self, task, png_bytes, meta=None):
        """Run one inference. Returns a JSON-serializable dict."""
        meta = meta or {}
        if task not in TASKS:
            raise InferenceError("INVALID_REQUEST", f"Unknown task '{task}'", 400)
        if self.manifest is None:
            raise InferenceError(
                "HEAD_UNAVAILABLE",
                f"Service not ready: {self.manifest_error}",
                409,
            )
        if self.encoder is None:
            raise InferenceError(
                "ENCODER_UNAVAILABLE",
                f"Encoder unavailable: {self.manifest_error}",
                409,
            )

        image = self._decode_image(png_bytes)
        expected_housing = meta.get("expected_housing")
        search_area = meta.get("search_area")
        self._validate_meta(search_area, image.size)
        self._validate_expected_housing(expected_housing)

        head = self.head_manager.heads[task]
        status = head.status()
        if status != "ok":
            raise InferenceError(
                "HEAD_UNAVAILABLE",
                f"Head '{task}' unavailable ({status}): {head.spec.path}",
                409,
            )

        # ---- encoder pass (timed) ----
        t0 = time.perf_counter()
        try:
            features = self.encoder.infer(image)
        except Exception as e:
            log.exception("Encoder inference failed")
            raise InferenceError("INFERENCE_ERROR", f"Encoder failed: {e}", 500) from e
        t_encoder = time.perf_counter() - t0

        # ---- hand-off (timed) ----
        t1 = time.perf_counter()
        features_copy = np.ascontiguousarray(features, dtype=np.float32)
        t_handoff = time.perf_counter() - t1

        # ---- head pass (timed) ----
        t2 = time.perf_counter()
        try:
            outputs = self.head_manager.infer(task, features_copy)
        except HeadLoadError as e:
            raise InferenceError("INFERENCE_ERROR", str(e), 500) from e
        t_head = time.perf_counter() - t2

        log.info(
            "task=%s encoder=%.1fms head=%.1fms handoff=%.1fms (features=%s)",
            task, t_encoder * 1000, t_head * 1000, t_handoff * 1000, features_copy.shape,
        )

        result = self._build_result(task, outputs, image, expected_housing)
        result["timings_ms"] = {
            "encoder": round(t_encoder * 1000, 2),
            "head": round(t_head * 1000, 2),
            "handoff": round(t_handoff * 1000, 3),
        }
        result["model_version"] = head.spec.model_version
        return result

    # ---- helpers --------------------------------------------------------

    def _decode_image(self, png_bytes):
        try:
            image = Image.open(io.BytesIO(png_bytes))
            image.load()
        except Exception as e:
            raise InferenceError("INVALID_IMAGE", f"Could not decode PNG: {e}", 422) from e
        if image.width < 4 or image.height < 4:
            raise InferenceError("INVALID_IMAGE", "Image too small", 422)
        return image.convert("RGB")

    def _validate_expected_housing(self, expected_housing):
        if expected_housing is None:
            return
        if not (isinstance(expected_housing, dict)
                and "width_px" in expected_housing and "height_px" in expected_housing):
            raise InferenceError(
                "INVALID_REQUEST",
                "expected_housing must be {\"width_px\": ..., \"height_px\": ...}",
                400,
            )
        if float(expected_housing["width_px"]) <= 0 or float(expected_housing["height_px"]) <= 0:
            raise InferenceError("INVALID_REQUEST", "expected_housing dimensions must be positive", 400)

    def _validate_meta(self, search_area, image_size):
        if search_area is None:
            return
        if not (isinstance(search_area, list) and len(search_area) == 4
                and all(isinstance(v, (int, float)) for v in search_area)):
            raise InferenceError("INVALID_REQUEST", "search_area must be [x, y, w, h]", 400)
        x, y, w, h = search_area
        width, height = image_size
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise InferenceError("INVALID_REQUEST", "search_area out of bounds", 400)

    def _build_result(self, task, outputs, image, expected_housing):
        width, height = image.size
        confidence = clamp01(outputs.get("confidence", 0.0))
        status = status_for(confidence)

        result = {
            "task": task,
            "status": status,
            "confidence": confidence,
        }

        if task == "bottom-vision":
            center_x = clamp01(outputs.get("center_x", 0.5)) * width
            center_y = clamp01(outputs.get("center_y", 0.5)) * height
            size_w = clamp01(outputs.get("size_w", 0.5)) * width
            size_h = clamp01(outputs.get("size_h", 0.5)) * height
            angle = clamp_angle(outputs.get("angle_deg", 0.0))
            result.update({
                "center_px": [round(center_x, 2), round(center_y, 2)],
                "size_px": [round(size_w, 2), round(size_h, 2)],
                "angle_deg": round(angle, 2),
                "ambiguity": "none",
            })
        elif task == "homing":
            visibility = clamp01(outputs.get("visibility", confidence))
            center_x = clamp01(outputs.get("center_x", 0.5)) * width
            center_y = clamp01(outputs.get("center_y", 0.5)) * height
            result.update({
                "visibility": visibility,
                "confidence": confidence,
                "center_px": [round(center_x, 2), round(center_y, 2)],
            })
            status = status_for(visibility)
            result["status"] = status
        else:
            raise InferenceError("INVALID_REQUEST", f"Unknown task '{task}'", 400)

        if expected_housing is not None:
            if not (isinstance(expected_housing, dict)
                    and "width_px" in expected_housing and "height_px" in expected_housing):
                raise InferenceError(
                    "INVALID_REQUEST",
                    "expected_housing must be {\"width_px\": ..., \"height_px\": ...}",
                    400,
                )
            result["expected_housing"] = {
                "width_px": float(expected_housing["width_px"]),
                "height_px": float(expected_housing["height_px"]),
            }

        # Enforce the confidence-derived status is only DETECTED when there
        # is a usable pose.
        if result["status"] != "DETECTED":
            # Non-detected results carry no usable pose fields; drop them.
            if task == "bottom-vision":
                result.pop("center_px", None)
                result.pop("size_px", None)
                result.pop("angle_deg", None)
                result.pop("ambiguity", None)
            else:
                result.pop("center_px", None)

        return result

    def infer_from_json_body(self, body_bytes):
        """Decode a /v1/infer request body and run the inference."""
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception as e:
            raise InferenceError("INVALID_REQUEST", f"Request body is not valid JSON: {e}", 400) from e
        if not isinstance(body, dict):
            raise InferenceError("INVALID_REQUEST", "Request body must be a JSON object", 400)

        task = body.get("task")
        if not isinstance(task, str) or not task:
            raise InferenceError("INVALID_REQUEST", "Missing 'task'", 400)

        image_b64 = body.get("image_png_base64")
        if not isinstance(image_b64, str) or not image_b64:
            raise InferenceError("INVALID_REQUEST", "Missing 'image_png_base64'", 400)
        try:
            png_bytes = base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise InferenceError("INVALID_IMAGE", f"Invalid base64 image: {e}", 422) from e

        meta = {
            "expected_housing": body.get("expected_housing"),
            "search_area": body.get("search_area"),
        }
        return self.infer(task, png_bytes, meta)
