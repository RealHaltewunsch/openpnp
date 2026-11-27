# ONNX Vision Wiring — Completion Log

Status date: 2026-11-27. Repository: `/Users/julius/openpnp`.
The integration and final clean-build verification are complete.
No push is part of this work.

## Starting point

Commits `24232d4be` and `b3de186f4` supplied the Python service, contract,
`BottomVisionEngine` and `/v1/infer` client. The previous checkpoint left
uncommitted settings/wizard hooks and the `OnnxBottomVisionSettings` classes.
This change completes that integration rather than extending model training.

## Completed wiring

- Vision Settings has a translated **ONNX Bottom Vision** filter and creation
  action. The classic Bottom Vision filter excludes ONNX entries. The parts
  selector already includes the subclass; choosing it while the service is
  disabled shows the enable-first message.
- `usePipeline()` hides pipeline controls and their bindings for ONNX.
  Specialization preserves the concrete engine. Both part and machine property
  sheets obtain the wizard through the settings type. Resetting settings across
  engine types uses the appropriate defaults.
- ONNX settings copy alignment options without copying pipeline parameters or
  stages. XML round-trips preserve their type and omit the unused pipeline;
  classic settings retain their pipeline serialization.
- `ReferenceBottomVision` uses `/v1/infer` poses for actual offsets when ONNX
  settings are assigned. Each shot moves the nozzle using the existing safe-Z
  rule, captures a settled image and always releases capture lighting. The
  existing pre/post-rotation calculations, iterative passes, vision offsets,
  part-size/offset checks, shot compositing and result display remain in use.
  Non-DETECTED responses and HTTP failures abort alignment without fallback.
- The shadow observer, its fields/accessors and its tests are deleted. Previously
  saved optional shadow attributes are tolerated and disappear on the next save.
- The Machine menu has a persistent service checkbox, startup on application
  launch when enabled, running-state display and an installation dialog. Startup
  and polling run outside the UI thread; shutdown terminates the owned process.
- The dialog selects Python, service directory, model/weights directory and port.
  Dependency installation creates `<service-dir>/.venv` and installs onnxruntime,
  numpy, Pillow and onnx with visible output. Successful service startup enables
  the checkbox. `.venv` and runtime logs are excluded from Git.
- Preferences use `org.openpnp.vision.service`; defaults are
  `<working-directory>/vision-service`, its `models/synthetic` subdirectory and
  loopback port 8010. Startup rejects occupied ports, checks contract/head health
  for up to 30 seconds and cleans up a failed process. Inference has a 10-second
  timeout. Runtime output goes to `<service-dir>/vision-service.log`.
- English/German strings, the engine plan, service contract and README are updated.

## Verification

- Python: `python3 -m unittest discover -s tests` — 28 passed.
- Live Python service, CPU provider: `/v1/health` returned `ok`; `/v1/infer`
  returned `DETECTED` with the bundled fixture geometry. The smoke-test process
  was terminated afterwards.
- New Java tests cover settings copy/reset/specialization, XML round-trips,
  type-aware wizards/pipeline visibility, legacy configuration loading,
  actual pre/post-rotation offsets using simulated capture and a mock HTTP
  service, rejection of uncertain detections, isolated preferences, health
  validation, startup/shutdown, occupied ports and failed child processes.
- Java with Temurin 21: clean compilation and 13 focused tests passed, followed
  by `mvn -q -B package` with the complete suite: **267 tests passed, zero failures,
  errors or skips**. The default newer JDK is unsuitable for the existing runner.
- The packaged JAR contains the ONNX settings/controller/dialog classes and no
  retired `ShadowBottomVisionObserver` class. Final diff/whitespace checks passed.
- This completion is committed locally; no push was performed.

## Remaining model/platform validation

The synthetic models are fixtures, not trained part detectors. Production use
requires independently validated real weights and real-machine placement checks.
Windows/D700, provider comparisons on trained models, and an interactive
installation-dialog/pip walkthrough have not been verified in this change.
