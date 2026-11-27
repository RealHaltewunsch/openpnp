# OpenPnP Vision Service

A local, versioned ONNX-Runtime vision service for OpenPnP. One encoder is
loaded per process and shared by every task head; heads are swappable, lazy-loaded
and warmed before first time-critical use. The service talks to OpenPnP over
loopback HTTP — the Java side never links ONNX Runtime itself.

Select **ONNX Bottom Vision** in OpenPnP's Vision Settings to use this service
for actual placement offsets. Existing Bottom Vision settings continue to use
their OpenCV pipelines. The bundled synthetic weights test the integration;
they are not a trained part detector and must not be used for production placement.

## Verified state (macOS arm64, this machine)

- onnxruntime **1.32.0** (CPU + CoreML), onnx 1.23.2, numpy, Pillow
- Provider auto-detect: `CoreMLExecutionProvider` on macOS arm64 (with CPU
  fallback for the few hand-crafted synthetic-encoder nodes CoreML does not
  support), `CPUExecutionProvider` elsewhere
- All 28 service tests pass (`python3 -m unittest discover -s tests -v`)
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

## OpenPnP integration

1. Open **Machine → Install Vision Service Dependencies...**. Select the service
   directory (containing `vision_service`), a Python executable, and a model package
   directory (containing `manifest.json` and its encoder/head weights). The defaults
   are `<working-directory>/vision-service` and its `models/synthetic` subdirectory;
   the default loopback port is 8010.
2. **Install dependencies** creates `.venv` inside the service directory and installs
   onnxruntime, numpy, Pillow and onnx there. It requires Python 3.9+, pip/venv and
   network access. Existing system Python packages are not modified. Installation
   output remains visible in the dialog.
3. **Start service** starts the configured model package and enables the Machine
   menu checkbox after health checks pass. The checkbox starts/stops the owned
   process and, when enabled, starts it again with OpenPnP. Stop the service before
   changing its directories, Python executable or port. Runtime output is appended
   to `vision-service.log` in the service directory.
4. In **Vision Settings**, choose **ONNX Bottom Vision**, add settings and assign
   them to the part/package or machine. Shared rotation, size and offset settings
   remain available; the OpenCV pipeline editor is hidden. If the service option
   is disabled, selecting ONNX shows a reminder to enable it first.

`LocalBottomVisionClient` calls `POST /v1/infer` for each settled, illuminated
camera shot. The returned pose drives the existing pre/post-rotation correction,
size/tolerance checks and vision compositing. Non-detected/uncertain results and
service failures stop alignment with an error; they do not use an OpenCV fallback.
The former shadow observer and its configuration fields have been removed.

Startup allows 30 seconds for a compatible health response with an available
bottom-vision head. A port already in use is rejected, and a failed owned process
is stopped. Other missing task heads do not prevent Bottom Vision from starting.
The first inference lazily loads and warms the head, with a 10-second request limit.

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
