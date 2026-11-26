"""Task heads.

A head is a small ONNX model bound to an encoder version and preprocessing.
Heads are loaded on demand (first use), warmed with a dummy inference before
first time-critical use, and reused for every subsequent request on the same
task. Heads never reload the encoder.
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)


class HeadLoadError(RuntimeError):
    """Raised when a head cannot be loaded or is incompatible with the encoder."""


class Head:
    def __init__(self, spec, encoder_version, provider_preference=None):
        self.spec = spec
        self.task = spec.name
        self.encoder_version = encoder_version
        self.provider_preference = provider_preference or ["CPUExecutionProvider"]
        self.session = None
        self.warmed = False
        self.session_id = None

    @property
    def version(self):
        return self.spec.version

    @property
    def model_version(self):
        return self.spec.model_version

    def status(self):
        """Health status for this head: ok | missing_weights | incompatible | load_error."""
        if not self.spec.path.is_file():
            return "missing_weights"
        if self.spec.encoder_version != self.encoder_version:
            return "incompatible"
        try:
            self._load()
        except Exception:
            return "load_error"
        return "ok"

    def _load(self):
        if self.session is not None:
            return
        if not self.spec.path.is_file():
            raise HeadLoadError(f"Head model not found: {self.spec.path}")
        if self.spec.encoder_version != self.encoder_version:
            raise HeadLoadError(
                f"Head '{self.task}' requires encoder version "
                f"'{self.spec.encoder_version}', loaded encoder is "
                f"'{self.encoder_version}'"
            )
        # Reuse the encoder's provider preference: heads run wherever the
        # encoder runs, so the feature hand-off is measured on the same
        # provider as the encoder.
        try:
            self.session = ort.InferenceSession(
                str(self.spec.path), providers=self.provider_preference
            )
        except Exception:
            self.session = ort.InferenceSession(
                str(self.spec.path), providers=["CPUExecutionProvider"]
            )
        self.session_id = id(self.session)
        log.info("Head '%s' loaded: version=%s", self.task, self.version)

    def warmup(self):
        """Load the head and run one dummy inference. Called before first use."""
        if self.warmed:
            return
        self._load()
        dim = int(self.spec.input.shape[-1])
        dummy = np.zeros((1, dim), dtype=np.float32)
        input_name = self.session.get_inputs()[0].name
        self.session.run(None, {input_name: dummy})
        self.warmed = True
        log.info("Head '%s' warmed", self.task)

    def infer(self, features):
        """Run the head on encoder features (numpy array [1, F]).

        Returns a dict of normalized outputs keyed by output_layout field
        name. Raises HeadLoadError if the head cannot run.
        """
        if not self.warmed:
            self.warmup()
        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        input_name = self.session.get_inputs()[0].name
        try:
            outputs = self.session.run(None, {input_name: features})
        except Exception as e:
            raise HeadLoadError(f"Head '{self.task}' inference failed: {e}") from e
        raw = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if raw.shape[0] != len(self.spec.output_layout):
            raise HeadLoadError(
                f"Head '{self.task}' returned {raw.shape[0]} values, "
                f"expected {len(self.spec.output_layout)}"
            )
        return {name: float(raw[i]) for i, name in enumerate(self.spec.output_layout)}


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def clamp_angle(x):
    """Normalize an angle in degrees to [-180, 180)."""
    x = float(x) % 360.0
    if x >= 180.0:
        x -= 360.0
    return x


def status_for(confidence):
    if confidence >= 0.6:
        return "DETECTED"
    if confidence >= 0.3:
        return "UNCERTAIN"
    return "NO_DETECTION"


class HeadManager:
    """Owns all heads for one service process. The encoder is shared."""

    def __init__(self, manifest, encoder):
        self.manifest = manifest
        self.encoder = encoder
        preference = list(getattr(encoder, "preference", ["CPUExecutionProvider"]))
        self.heads = {
            task: Head(spec, encoder.spec.version, preference)
            for task, spec in manifest.heads.items()
        }

    def tasks(self):
        return sorted(self.heads.keys())

    def get(self, task):
        head = self.heads.get(task)
        if head is None:
            raise KeyError(task)
        return head

    def health(self):
        out = {}
        for task, head in self.heads.items():
            out[task] = {
                "version": head.version,
                "encoder_version": head.spec.encoder_version,
                "status": head.status(),
                "warmed": head.warmed,
            }
        return out

    def infer(self, task, features):
        head = self.get(task)
        status = head.status()
        if status != "ok":
            raise HeadLoadError(f"Head '{task}' unavailable ({status}): {head.spec.path}")
        return head.infer(features)
