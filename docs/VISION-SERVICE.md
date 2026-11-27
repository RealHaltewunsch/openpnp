# Vision-Service Contract (Versioned Loopback Service)

This document is the contract for the local vision service. It is the
specification the service implementation, the OpenPnP client and the synthetic
test models are built against. Any change to the contract is a breaking
change and must bump `contract_version`.

## Scope

- Loopback only: the service binds to `127.0.0.1` and is not reachable from
  the network.
- One process loads the shared encoder **once**. Task heads are loaded on
  demand, warmed before first time-critical use, and share the encoder
  session across tasks.
- A new encoder-weight set is a new model version. Heads are bound to an
  encoder version and the preprocessing of that encoder.
- ONNX Runtime CPU is the cross-platform reference (Windows x64, macOS arm64).
  DirectML (Windows) and CoreML (Apple Silicon) are hardware-verified
  providers, not the baseline.

## Model manifest

`<models-dir>/manifest.json` describes one encoder and all heads:

```json
{
  "version": 1,
  "contract_version": 1,
  "encoder": {
    "name": "synthetic-encoder",
    "version": "synthetic-1.0.0",
    "path": "encoder.onnx",
    "input":  { "name": "image", "shape": [1, 3, 224, 224], "dtype": "float32" },
    "output": { "name": "features", "shape": [1, 16], "dtype": "float32" },
    "preprocessing": {
      "resize": [224, 224],
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225],
      "layout": "CHW",
      "channel_first": true
    }
  },
  "heads": {
    "bottom-vision": {
      "version": "synthetic-1.0.0",
      "path": "heads/bottom-vision.onnx",
      "encoder_version": "synthetic-1.0.0",
      "input":  { "name": "features", "shape": [1, 16], "dtype": "float32" },
      "output": { "name": "result", "shape": [1, 6], "dtype": "float32" },
      "output_layout": ["center_x", "center_y", "size_w", "size_h", "angle_deg", "confidence"]
    },
    "homing": {
      "version": "synthetic-1.0.0",
      "path": "heads/homing.onnx",
      "encoder_version": "synthetic-1.0.0",
      "input":  { "name": "features", "shape": [1, 16], "dtype": "float32" },
      "output": { "name": "result", "shape": [1, 4], "dtype": "float32" },
      "output_layout": ["center_x", "center_y", "visibility", "confidence"]
    }
  }
}
```

- `contract_version`: contract version this manifest targets (currently `1`).
  A manifest with a higher or unknown `contract_version` is rejected at
  startup.
- `encoder.path`, `heads.<task>.path`: relative to the manifest directory.
- Head outputs are **normalized**: `center_x`/`center_y`/`size_w`/`size_h`
  are fractions of the input image width/height in `[0, 1]`; `angle_deg` is in
  `[-180, 180]`; `confidence`/`visibility` are in `[0, 1]`. The service
  denormalizes to original image pixels.
- `encoder_version` in a head must equal the loaded encoder version, otherwise
  the head is reported `incompatible`.
- `ambiguity` (bottom-vision only) is not an output tensor; the service sets
  it from the head's output (see below).

## Health endpoint

`GET /v1/health` — always returns HTTP 200.

```json
{
  "status": "ok" | "degraded",
  "service": "openpnp-vision",
  "version": "0.1.0",
  "contract_version": 1,
  "onnx_runtime_version": "1.32.0",
  "provider": "CoreMLExecutionProvider",
  "providers_available": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
  "encoder_version": "synthetic-1.0.0",
  "encoder_session_id": "9f2b0d4e-...",
  "encoder_sessions_created": 1,
  "heads": {
    "bottom-vision": { "version": "synthetic-1.0.0", "encoder_version": "synthetic-1.0.0", "status": "ok", "warmed": false },
    "homing":        { "version": "synthetic-1.0.0", "encoder_version": "synthetic-1.0.0", "status": "missing_weights", "warmed": false }
  }
}
```

- `status` is `degraded` if the encoder is missing/incompatible or any head
  is not `ok`.
- Per-head `status`: `ok`, `missing_weights`, `incompatible`,
  `load_error`. `encoder_version` mismatch ⇒ `incompatible`.
- `encoder_session_id` is stable across task switches; the test for same-
  encoder reuse asserts it does not change between two different tasks.
- `encoder_sessions_created` increments when a new encoder session is created
  (expected exactly `1` per process).

## Inference route

`POST /v1/infer` — one route, task selected by the body.

Request body (JSON):

```json
{
  "task": "bottom-vision",
  "image_png_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "expected_housing": { "width_px": 120, "height_px": 60 },
  "search_area": [0, 0, 640, 480]
}
```

- `task`: `bottom-vision` | `homing` (others rejected with 400).
- `image_png_base64`: PNG, original pixels.
- `expected_housing` (optional): expected housing size in original pixels;
  used for ambiguity reporting, not for inference.
- `search_area` (optional): `[x, y, w, h]` in original pixels; the head output
  is expected inside this area. Used for validation, not for inference.

Response (HTTP 200):

```json
{
  "task": "bottom-vision",
  "status": "DETECTED" | "NO_DETECTION" | "UNCERTAIN" | "ERROR",
  "confidence": 0.87,
  "model_version": "synthetic-1.0.0",
  "center_px": [320.0, 240.0],
  "size_px": [120.0, 60.0],
  "angle_deg": 12.5,
  "ambiguity": "none" | "mirror" | "180" | "symmetric"
}
```

- Homing response: `{ task, status, confidence, visibility, center_px, model_version }`
  where `visibility` is in `[0, 1]`. No `size_px`/`angle_deg`/`ambiguity`.
- Status vocabulary: `DETECTED` (confidence/visibility ≥ 0.6), `UNCERTAIN`
  (0.3–0.6), `NO_DETECTION` (< 0.3), `ERROR` (inference failure).
- `ambiguity` for bottom-vision: `none` (default), `mirror` (housing has
  reflective symmetry — angle is valid modulo 180°), `180` (180°-symmetric
  housing), `symmetric` (square-like housing, angle valid modulo 90°). The
  service reports `none` for the synthetic models; trained heads will report
  a real value.
- Errors (`ERROR` or non-200): `{ "task", "status": "ERROR", "error": { "code": "...", "detail": "..." } }`
  with `code` one of `INVALID_REQUEST`, `HEAD_UNAVAILABLE`, `INVALID_IMAGE`,
  `INFERENCE_ERROR`. HTTP 400/409/422/500 respectively.

## Error model

- Missing encoder weights ⇒ health `degraded`, inference HTTP 409
  `HEAD_UNAVAILABLE` (or `ENCODER_UNAVAILABLE` for the encoder itself).
- Head missing weights ⇒ health shows head `missing_weights`, inference
  HTTP 409 `HEAD_UNAVAILABLE` with the missing path in `detail`.
- Head/encoder version mismatch ⇒ head `incompatible`, inference HTTP 409
  `HEAD_UNAVAILABLE`.
- Invalid image ⇒ HTTP 422 `INVALID_IMAGE`.
- Bad request shape ⇒ HTTP 400 `INVALID_REQUEST`.
- Inference failure ⇒ HTTP 500 `INFERENCE_ERROR`.

## Versioning

- `contract_version` in the manifest and health response. The service refuses
  to load a manifest with an unknown `contract_version`.
- `model_version` in every inference response identifies the encoder/head
  pair that produced the result (`encoder_version` + head `version`, joined
  by `+`).

## Example requests and responses

Bottom-vision inference (synthetic models):

```
POST /v1/infer
{"task":"bottom-vision","image_png_base64":"...","expected_housing":{"width_px":120,"height_px":60}}
```

```json
{"task":"bottom-vision","status":"DETECTED","confidence":0.9,
 "model_version":"synthetic-1.0.0+synthetic-1.0.0",
 "center_px":[320.0,240.0],"size_px":[120.0,60.0],"angle_deg":12.5,"ambiguity":"none"}
```

Homing inference (synthetic models):

```
POST /v1/infer
{"task":"homing","image_png_base64":"..."}
```

```json
{"task":"homing","status":"DETECTED","confidence":0.9,"visibility":0.9,
 "center_px":[320.0,240.0],"model_version":"synthetic-1.0.0+synthetic-1.0.0"}
```

## OpenPnP engine wiring (2026-11-27)

`OnnxBottomVisionSettings` selects this service as the actual Bottom Vision
engine. It stores shared alignment options without an OpenCV pipeline. Existing
`BottomVisionSettings` retain their pipeline behavior. Type-aware wizards and
specialization preserve the selected engine, including XML round-trips.

The Machine menu owns service startup/shutdown and an installation dialog with
Python, service directory, model directory and port selection. Dependencies are
installed into `<service-dir>/.venv`. Preferences are stored under
`org.openpnp.vision.service`; defaults are port 8010 and
`<working-directory>/vision-service/models/synthetic`. See the service README
for setup and startup diagnostics.

For ONNX settings, `ReferenceBottomVision` captures each settled shot with light
cleanup in a `finally` block, calls `/v1/infer`, and converts the detected image
pose into `RotatedRect` geometry. The existing camera conversion, pre-rotation
passes, compositing, size/tolerance checks and result display consume this pose.
OpenCV stages are not executed. Any non-DETECTED status or request failure aborts
alignment. The shadow observer has been removed.

The supplied synthetic package validates service plumbing and returns fixture
geometry. Real placement requires trained and independently validated weights.
Windows startup/provider behavior and real-machine placement remain unverified.
