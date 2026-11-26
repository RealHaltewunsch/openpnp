# OpenPnP Vision Service

A local, versioned ONNX-Runtime vision service for OpenPnP. One encoder is
loaded per process and shared by every task head; heads are swappable, lazy-loaded
and warmed before first time-critical use. The service talks to OpenPnP over
loopback HTTP — the Java side never links ONNX Runtime itself.

This service is in **shadow mode**: it runs alongside the existing OpenPnP
bottom-vision path and only logs what it would see. The existing pipeline is
unchanged until the service is accepted.

## Verified state (macOS arm64, this machine)

- onnxruntime **1.32.0** (CPU + CoreML), onnx 1.23.2, numpy, Pillow
- Provider auto-detect: `CoreMLExecutionProvider` on macOS arm64 (with CPU
  fallback for the few hand-crafted synthetic-encoder nodes CoreML does not
  support), `CPUExecutionProvider` elsewhere
- All 29 service tests pass (`python3 -m unittest discover -s tests -v`)
- Live smoke test: `/v1/health` and `/v1/infer`
  route all return the expected results on the fixture image
- Startup via `./start-mac.sh` verified on macOS arm64

**Not yet verified:** Windows x64 (DirectML on D700), real trained DINOv3
weights, and DirectML/CoreML head-to-head latency/deviation on trained models.
The Windows start script is provided but **untested**.

## Layout

```
vision_service/        the service (stdlib http.server, no web framework)
  __main__.py          CLI: --models-dir --host --port --provider --log-level
  server.py            routes: GET /v1/health, POST /v1/infer
  service.py           VisionService: manifest+encoder load, health, inference
  encoder.py           Encoder: once-loaded session, preprocess, provider detect
  heads.py             Head/HeadManager: lazy load, warmup, status, reuse
  manifest.py          manifest loading + validation
scripts/generate_synthetic.py   test-only synthetic model generator
tests/test_service.py           full test suite (unittest)
tests/images/fixture.png        saved fixture image (+ fixture.json with truth)
start-mac.sh                    macOS start script (verified)
start-windows.ps1               Windows start script (untested)
```

## Quickstart

```sh
cd vision-service

# 1. Install dependencies (Python 3.9+)
python3 -m pip install onnxruntime numpy Pillow onnx

# 2. Generate the synthetic test model package (TEST-ONLY, not DINOv3)
python3 scripts/generate_synthetic.py --out models/synthetic

# 3. Start the service
./start-mac.sh models/synthetic           # macOS (verified)
# powershell -File start-windows.ps1      # Windows (untested)

# 4. Check health
curl -s http://127.0.0.1:8010/v1/health | python3 -m json.tool
```

The service starts even if a head is missing — `/v1/health` reports the
problem and `/v1/infer` returns `409 HEAD_UNAVAILABLE` instead of crashing.

## Reproducible test invocation

From a clean checkout (no pre-generated models needed — the tests generate a
fresh synthetic package in a temp dir):

```sh
cd vision-service
python3 -m unittest discover -s tests -v
```

To reproduce a single end-to-end check against the saved fixture image:

```sh
./start-mac.sh models/synthetic &
curl -s -X POST http://127.0.0.1:8010/v1/infer \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"bottom-vision\",\"image_png_base64\":\"$(base64 -i tests/images/fixture.png | tr -d '\n')\"}"
kill %1
```

Expected: `status: DETECTED`, `confidence ≈ 0.9`, center ≈ (160, 120) px,
size ≈ (176, 120) px, angle ≈ 10–12°.

## Model package

A model package is a directory containing `manifest.json`, the encoder ONNX
file and one ONNX file per task head. See `docs/VISION-SERVICE.md` for the
full contract (manifest format, health semantics, routes, error model).

Encoder version + head version + feature layout + preprocessing are bound
together in the manifest; changing the encoder weights means a new model
version, and a head bound to the old encoder version reports `incompatible`.

The manifest lists missing *files* as per-head/per-encoder status (never at
load time), so the service can start and report through `/v1/health`.

## Synthetic models (test-only)

`scripts/generate_synthetic.py` produces a deterministic hand-crafted encoder
(spatial statistics over the image: mean RGB, luminance stats, dark-region
fraction/extent/center, center-of-mass, min/max luminance → 16 features) and
linear heads fit to known-good renders. This is **not** a trained DINOv3 model
and does not pretend to be one — it exists to prove the service plumbing:
load-once encoder, shared session, head warmup, task switching, normalized
output handling, and the error/status model.

Known limitation: the linear head predicts center/size well (mean abs error
~0.3–0.8% of the image) but only roughly recovers angle (~14° mean error),
because the synthetic encoder's features are rotation-invariant and angle is
a nonlinear function of dark-region shape. The fixture renders to ~10–12° on
a 12° target. This is expected; the fixture tolerance in the tests reflects it.

## Task heads

| Task | Outputs (normalized) | Meaning |
|------|----------------------|---------|
| `bottom-vision` | center_x, center_y, size_w, size_h, angle_deg, confidence | Bottom pose, fractions [0,1], angle [-180,180] |
| `homing` | center_x, center_y, visibility, confidence | Homing target, fractions [0,1] |

Responses denormalize to original pixels (`center_px`, `size_px`) and report
`model_version` = `<encoder_version>+<head_version>`. `ambiguity` is reported
as `none` for synthetic heads; a trained head may report `mirror`, `180` or
`symmetric`.

## Status codes

- `DETECTED` (conf ≥ 0.6) — usable pose/visibility
- `UNCERTAIN` (0.3–0.6) — present but not trusted; pose fields dropped
- `NO_DETECTION` (< 0.3) — nothing usable; pose fields dropped
- `ERROR` — invalid request, missing head, inference failure

Errors are JSON `{"status": "ERROR", "error": {"code": ..., "detail": ...}}`
with HTTP 400 (INVALID_REQUEST), 409 (HEAD_UNAVAILABLE), 422 (INVALID_IMAGE)
or 500 (INFERENCE_ERROR). `/v1/health` always returns HTTP 200 with per-head
status `ok` / `missing_weights` / `incompatible` / `load_error`.

## OpenPnP integration (shadow mode)

The existing `LocalBottomVisionClient` talks to `POST /v1/infer` (JSON body).
OpenPnP's new shadow-mode observer reads the same results and logs them
alongside the existing correction path without changing it. See the Java
classes under `org.openpnp.vision.alternative` and `ReferenceBottomVision`.

## Start commands

macOS arm64 (verified):

```sh
./start-mac.sh models/synthetic
# or: python3 -m vision_service --models-dir models/synthetic --port 8010
```

Windows x64 (untested — macOS is the current development target):

```powershell
powershell -File start-windows.ps1 -ModelsDir models/synthetic
# or: python -m vision_service --models-dir models/synthetic --port 8010
```

Both bind loopback only (`127.0.0.1`). Force a provider with
`--provider CoreMLExecutionProvider` / `--provider CPUExecutionProvider` /
`--provider DmlExecutionProvider` (Windows). Default is auto: platform
accelerator with CPU fallback.
