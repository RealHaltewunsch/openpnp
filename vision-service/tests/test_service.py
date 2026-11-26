#!/usr/bin/env python3
"""Tests for the synthetic vision service.

Covers: health, inference (bottom-vision + homing), task switching with a
single shared encoder session, warmup reporting, missing-weight status,
manifest validation and error paths — against a freshly generated synthetic
model package, so it works from a clean checkout.

Run:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import onnxruntime  # noqa: F401  (import check)

from vision_service.service import InferenceError, VisionService
from vision_service.manifest import ManifestError, load_manifest
from vision_service.server import create_server
from vision_service.heads import status_for, clamp01, clamp_angle

FIXTURE_PNG = ROOT / "tests" / "images" / "fixture.png"
FIXTURE_JSON = ROOT / "tests" / "images" / "fixture.json"


def run_generator(models_dir):
    """Generate the synthetic model package into models_dir."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_synthetic.py"), "--out", str(models_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Synthetic generator failed:\n{result.stdout}\n{result.stderr}")


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServiceTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp(prefix="vision-test-"))
        cls.models_dir = cls._tmp / "models"
        run_generator(cls.models_dir)
        cls.service = VisionService(cls.models_dir, provider="CPUExecutionProvider")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def png_bytes(self):
        return FIXTURE_PNG.read_bytes()

    def png_b64(self):
        return base64.b64encode(self.png_bytes()).decode("ascii")


class HealthTest(ServiceTestBase):
    def test_health_is_ok(self):
        health = self.service.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["service"], "openpnp-vision")
        self.assertEqual(health["contract_version"], 1)
        self.assertEqual(health["encoder_sessions_created"], 1)
        self.assertTrue(health["encoder_session_id"])
        self.assertEqual(health["provider"], "CPUExecutionProvider")
        self.assertEqual(health["encoder_version"], "synthetic-1.0.0")
        self.assertEqual(set(health["heads"]), {"bottom-vision", "homing"})
        for head in health["heads"].values():
            self.assertEqual(head["status"], "ok")
            self.assertFalse(head["warmed"])

    def test_health_reports_provider_and_sessions(self):
        health = self.service.health()
        self.assertIn("CPUExecutionProvider", health["providers_available"])
        self.assertEqual(health["encoder_sessions_created"], 1)


class InferenceTest(ServiceTestBase):
    def test_bottom_vision_fixture(self):
        result = self.service.infer("bottom-vision", self.png_bytes(), {})
        self.assertEqual(result["task"], "bottom-vision")
        self.assertEqual(result["status"], "DETECTED")
        self.assertGreater(result["confidence"], 0.6)

        fixture = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
        exp = fixture["expected"]
        self.assertAlmostEqual(result["center_px"][0], exp["center_px"][0], delta=16.0)
        self.assertAlmostEqual(result["center_px"][1], exp["center_px"][1], delta=16.0)
        self.assertAlmostEqual(result["size_px"][0], exp["size_px"][0], delta=16.0)
        self.assertAlmostEqual(result["size_px"][1], exp["size_px"][1], delta=16.0)
        self.assertAlmostEqual(result["angle_deg"], exp["angle_deg"], delta=25.0)
        self.assertEqual(result["ambiguity"], "none")
        self.assertEqual(result["model_version"], "synthetic-1.0.0+synthetic-1.0.0")
        self.assertIn("timings_ms", result)

    def test_homing_fixture(self):
        result = self.service.infer("homing", self.png_bytes(), {})
        self.assertEqual(result["task"], "homing")
        self.assertEqual(result["status"], "DETECTED")
        self.assertIn("visibility", result)
        self.assertIn("center_px", result)
        self.assertGreater(result["visibility"], 0.6)

    def test_head_switch_reuses_one_encoder_session(self):
        session_id_before = self.service.encoder.session_id
        n_heads_before = len(self.service.head_manager.heads)
        r1 = self.service.infer("bottom-vision", self.png_bytes(), {})
        r2 = self.service.infer("homing", self.png_bytes(), {})
        self.assertEqual(self.service.encoder.session_id, session_id_before)
        self.assertEqual(self.service.encoder.session_id, session_id_before)
        # Both heads warmed, both still share the same encoder session.
        self.assertTrue(self.service.head_manager.heads["bottom-vision"].warmed)
        self.assertTrue(self.service.head_manager.heads["homing"].warmed)
        self.assertEqual(n_heads_before, 2)
        # Responses carry the model version of the head that ran.
        self.assertEqual(r1["model_version"], "synthetic-1.0.0+synthetic-1.0.0")
        self.assertEqual(r2["model_version"], "synthetic-1.0.0+synthetic-1.0.0")

    def test_warmup_reports_warmed_after_inference(self):
        health = self.service.health()
        for head in health["heads"].values():
            self.assertTrue(head["warmed"])

    def test_expected_housing_echoed(self):
        result = self.service.infer(
            "bottom-vision", self.png_bytes(),
            {"expected_housing": {"width_px": 176.0, "height_px": 120.0}},
        )
        self.assertEqual(result["expected_housing"], {"width_px": 176.0, "height_px": 120.0})


class ErrorPathTest(ServiceTestBase):
    def test_unknown_task_rejected(self):
        with self.assertRaises(InferenceError) as ctx:
            self.service.infer("not-a-task", self.png_bytes(), {})
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_invalid_image_rejected(self):
        with self.assertRaises(InferenceError) as ctx:
            self.service.infer("bottom-vision", b"not a png", {})
        self.assertEqual(ctx.exception.code, "INVALID_IMAGE")
        self.assertEqual(ctx.exception.status_code, 422)

    def test_invalid_json_body_rejected(self):
        with self.assertRaises(InferenceError) as ctx:
            self.service.infer_from_json_body(b"{not json")
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")

    def test_missing_task_rejected(self):
        with self.assertRaises(InferenceError) as ctx:
            self.service.infer_from_json_body(json.dumps({"image_png_base64": self.png_b64()}).encode())
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")

    def test_missing_image_rejected(self):
        with self.assertRaises(InferenceError) as ctx:
            self.service.infer_from_json_body(json.dumps({"task": "bottom-vision"}).encode())
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")

    def test_invalid_base64_rejected(self):
        with self.assertRaises(InferenceError) as ctx:
            self.service.infer_from_json_body(json.dumps({
                "task": "bottom-vision", "image_png_base64": "!!!not-base64!!!",
            }).encode())
        self.assertEqual(ctx.exception.code, "INVALID_IMAGE")

    def test_search_area_out_of_bounds_rejected(self):
        with self.assertRaises(InferenceError) as ctx:
            self.service.infer("bottom-vision", self.png_bytes(),
                               {"search_area": [0, 0, 99999, 99999]})
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")

    def test_bad_expected_housing_rejected(self):
        with self.assertRaises(InferenceError) as ctx:
            self.service.infer("bottom-vision", self.png_bytes(),
                               {"expected_housing": {"width_px": 0}})
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")

    def test_infer_from_json_body_success(self):
        result = self.service.infer_from_json_body(json.dumps({
            "task": "bottom-vision",
            "image_png_base64": self.png_b64(),
        }).encode())
        self.assertEqual(result["task"], "bottom-vision")
        self.assertEqual(result["status"], "DETECTED")


class HTTPServerTest(ServiceTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.port = free_port()
        cls.server = create_server("127.0.0.1", cls.port, cls.service)
        import threading
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        super().tearDownClass()

    def _get(self, path):
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def _post(self, path, body, headers=None):
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=body, method="POST",
            headers=headers or {},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health_route(self):
        status, body = self._get("/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["encoder_sessions_created"], 1)

    def test_infer_route(self):
        body = json.dumps({
            "task": "bottom-vision",
            "image_png_base64": self.png_b64(),
        }).encode()
        status, resp = self._post("/v1/infer", body, {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(resp["status"], "DETECTED")

    def test_legacy_bottom_vision_route_removed(self):
        status, resp = self._post(
            "/v1/bottom-vision", self.png_bytes(),
            {"Content-Type": "image/png"},
        )
        self.assertEqual(status, 404)
        self.assertEqual(resp["status"], "ERROR")

    def test_unknown_route(self):
        status, resp = self._get("/nope")
        self.assertEqual(status, 404)


class MissingWeightsTest(ServiceTestBase):
    def test_missing_head_file_reports_missing_weights(self):
        import json as _json
        models = self._tmp / "models-missing"
        models.mkdir()
        # Copy manifest + encoder, remove the bottom-vision head.
        shutil.copy(self.models_dir / "encoder.onnx", models / "encoder.onnx")
        shutil.copy(self.models_dir / "manifest.json", models / "manifest.json")
        (models / "heads").mkdir()
        shutil.copy(self.models_dir / "heads" / "homing.onnx", models / "heads" / "homing.onnx")
        service = VisionService(models, provider="CPUExecutionProvider")
        health = service.health()
        self.assertEqual(health["heads"]["bottom-vision"]["status"], "missing_weights")
        self.assertEqual(health["heads"]["homing"]["status"], "ok")
        self.assertEqual(health["status"], "degraded")
        with self.assertRaises(InferenceError) as ctx:
            service.infer("bottom-vision", self.png_bytes(), {})
        self.assertEqual(ctx.exception.code, "HEAD_UNAVAILABLE")


class ManifestValidationTest(unittest.TestCase):
    def test_valid_manifest_loads(self):
        manifest = load_manifest(self._models_dir() / "manifest.json")
        self.assertEqual(manifest.contract_version, 1)
        self.assertEqual(manifest.encoder.version, "synthetic-1.0.0")
        self.assertEqual(set(manifest.heads), {"bottom-vision", "homing"})

    def test_bad_contract_version_rejected(self):
        bad = self._write_bad_manifest({"contract_version": 999})
        with self.assertRaises(ManifestError):
            load_manifest(bad)

    def test_output_layout_mismatch_rejected(self):
        bad = self._write_bad_manifest(
            {"heads": {"bottom-vision": {"version": "x", "path": "heads/bottom-vision.onnx",
                                          "encoder_version": "synthetic-1.0.0",
                                          "input": {"name": "features", "shape": [1, 16]},
                                          "output": {"name": "result", "shape": [1, 5]},
                                          "output_layout": ["a", "b", "c", "d", "e", "f"]}}}
        )
        with self.assertRaises(ManifestError):
            load_manifest(bad)

    def test_missing_manifest_rejected(self):
        with self.assertRaises(ManifestError):
            load_manifest(Path("/nonexistent/manifest.json"))

    # helpers
    def _models_dir(self):
        tmp = Path(tempfile.mkdtemp(prefix="manifest-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        run_generator(tmp)
        return tmp

    def _write_bad_manifest(self, overrides):
        import copy
        tmp = Path(tempfile.mkdtemp(prefix="manifest-bad-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copy(self._models_dir() / "encoder.onnx", tmp / "encoder.onnx")
        shutil.copytree(self._models_dir() / "heads", tmp / "heads")
        raw = json.loads((self._models_dir() / "manifest.json").read_text(encoding="utf-8"))
        raw.update(overrides)
        if "heads" in overrides:
            raw["heads"] = overrides["heads"]
        (tmp / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
        return tmp / "manifest.json"


class PureHelperTest(unittest.TestCase):
    def test_status_for(self):
        self.assertEqual(status_for(0.9), "DETECTED")
        self.assertEqual(status_for(0.6), "DETECTED")
        self.assertEqual(status_for(0.5), "UNCERTAIN")
        self.assertEqual(status_for(0.3), "UNCERTAIN")
        self.assertEqual(status_for(0.1), "NO_DETECTION")

    def test_clamp01(self):
        self.assertEqual(clamp01(-1), 0.0)
        self.assertEqual(clamp01(0.5), 0.5)
        self.assertEqual(clamp01(2), 1.0)

    def test_clamp_angle(self):
        self.assertAlmostEqual(clamp_angle(190), -170.0)
        self.assertAlmostEqual(clamp_angle(-190), 170.0)
        self.assertAlmostEqual(clamp_angle(90), 90.0)


if __name__ == "__main__":
    unittest.main()
