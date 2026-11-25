# Bottom Vision (DINOv3) Integration & Build/Packaging — Learnings and Plan

**Status:** living document. Private, non-commercial learning use.
**Scope:** this fork. No code changes were made to the
application itself; the packaging tooling and pipeline below are the deliverable.

---

## 1. What was learned

### 1.1 Fork state

- The fork adds a **bottom-vision engine interface**, not a full integration:
  - `src/main/java/org/openpnp/vision/alternative/BottomVisionEngine.java` — a
    small interface: `Result analyze(BufferedImage image)` plus a `Result`/`Pose`
    model (image-space center, size, clockwise angle; the caller applies part
    symmetry and the camera-to-machine transform).
  - `src/main/java/org/openpnp/vision/alternative/LocalBottomVisionClient.java` —
    HTTP client. `POST http://127.0.0.1:<port>/v1/bottom-vision` with a PNG body;
    expects a JSON response with `status` (`DETECTED` | `NO_DETECTION` |
    `UNCERTAIN` | `ERROR`), optional `confidence` (0–1), and for detected results
    `center_px` and `size_px` (two-number arrays) and `angle_deg`.
  - `src/test/java/org/openpnp/vision/alternative/LocalBottomVisionClientTest.java`
    — tests for the client.
- **Neither is wired into the pipeline yet.** `ReferenceBottomVision` /
  `CvPipeline` still do the OpenCV-based bottom vision in-process. Nothing in
  the GUI/machine currently calls `LocalBottomVisionClient`.

### 1.2 Existing in-process vision

- `org.openpnp.machine.reference.vision.ReferenceBottomVision` →
  `vision/pipeline/CvPipeline` stages (`nu.pattern.OpenCV`, loaded by
  `OpenCvUtils.loadLocally()` from the bundled `org.openpnp:opencv:4.5.5-1` jar).
- The OpenCV jar bundles natives for linux / osx (ARMv8, x86_64) / windows
  (x86_32, x86_64); `loadLocally()` extracts them from the classpath. This is why
  a packaged app should run without a separate OpenCV install.
- Build: `mvn package` (Java 11 target, `org.openpnp.Main` main class,
  `lib/` classpath prefix; checkstyle is enforced and fails the build).
  Output: `target/openpnp-gui-0.0.1-alpha-SNAPSHOT.jar` + `target/lib/` (96 jars,
  ~180 MB).

### 1.3 Upstream build/deploy (original `openpnp/openpnp`)

- Builds/deploys macOS installers via **install4j** in
  `.github/workflows/build_and_deploy.yml`. It requires secrets and a commercial
  install4j license (`INSTALL4J_10_LICENSE_KEY`, code-signing and notarization
  passwords, S3 upload) and only runs on the upstream `main`/`test` branches.
  **The fork cannot reproduce it.**
- Test matrix: Java 11 / 17 / 19 on `ubuntu-latest`.
- No Windows build path exists in the upstream repo: no CI, no docs, no
  documented command. `OpenPnP.install4j` contains Windows_32/64 install targets
  and native paths (`windows/x86`, `windows/x64`), but they are built only via
  install4j + a license.
- Launchers: `openpnp.sh` (macOS/Linux), `openpnp.bat` (Windows). Both use
  `--add-opens=java.base/java.lang=ALL-UNNAMED`,
  `--add-opens=java.desktop/java.awt=ALL-UNNAMED`,
  `--add-opens=java.desktop/java.awt.color=ALL-UNNAMED`.

### 1.4 Packaging findings

- **jpackage** (free, ships with the JDK) is the only license-free way to build a
  native macOS `.app` from source. Verified on this machine (JDK 21 Temurin):
  built `dist/OpenPnP.app` (356 MB, includes a bundled JRE) with the fork's icon,
  and smoke-tested it: `./dist/OpenPnP.app/Contents/MacOS/OpenPnP --version` →
  `2.6_2026-11-25_12-07-26.d5eb7d1` (fork build, git-hash stamped).
- **Cross-platform packaging is not supported** by jpackage: it builds only for
  the OS it runs on. A Windows `.exe` cannot be built on macOS; it needs a
  Windows machine or a `windows-latest` GitHub Actions runner.
- Windows `.exe` installers additionally require **WiX 3.x** (WiX 5/6 break
  jpackage). `app-image` (unpackaged directory) does not require WiX.
- **Windows build verified** (2026-11-25, Windows 11 PC): Temurin JDK 21.0.13.1
  via winget, Maven 3.9.16 (manual, no winget package), WiX 3.14.1 (manual,
  `wix314-binaries.zip`), repo cloned locally. Built
  `dist/OpenPnP-2.6.0.exe` (~236 MB) with icon, plus a Windows `app-image` whose
  launcher was smoke-tested: `dist-appimage\OpenPnP\OpenPnP.exe --version` →
  `2.6_2026-11-25_15-01-05.7da4a57` (exit 0). A `.ico` was generated from the
  repo's iconset (`icon_512x512@2x.png` → multi-size ICO) and is committed at
  `src/main/resources/icons/OpenPnP.ico`.

---

## 2. Goals

1. Explore and integrate a **DINOv3-based bottom-vision engine** (Meta DINOv3).
2. **Per-task weights**: fiducials / marking / part pose each use their own
   checkpoint.
3. Keep it **private, non-commercial, learning-use only**.
4. Package the fork as:
   - **macOS `.app`** (done, verified — see §5)
   - **Windows `.exe`** (needs a Windows host/runner — see §5.2)
5. Add a release pipeline that builds both from a release build.

---

## 3. DINOv3 integration plan (Option A: external Python process)

### 3.1 Why Option A

The fork's `BottomVisionEngine` is deliberately **process-agnostic**: it consumes
a `BufferedImage` and returns a pose. The fork's existing client
(`LocalBottomVisionClient`) speaks HTTP on `127.0.0.1` — exactly the seam where a
Python/torch process can plug in.

In-JVM integration (OpenCV-style, or Java-side inference) was rejected:

- DINOv3 checkpoints are `.pt` (PyTorch) files; they cannot be loaded by a
  Java classpath dependency. Java options (ONNX export + ONNX Runtime Java, DJL,
  JNI/native) are all heavy, and ONNX export of DINOv3 + head training is a
  larger, separate project.
- The Python process can be trained/fine-tuned per task with PyTorch and simply
  expose an HTTP endpoint. The Java side stays small and stays portable.

### 3.2 Contract

The endpoint (already the fork's client contract):

- `POST /v1/bottom-vision` with `Content-Type: image/png` body → response JSON:
  - `status`: `DETECTED` | `NO_DETECTION` | `UNCERTAIN` | `ERROR`
  - `confidence`: optional 0–1
  - `DETECTED`: `center_px`: [x, y], `size_px`: [w, h], `angle_deg`: number
  - Non-detected: no pose fields

### 3.3 Per-task weights

Checkpoints are **not committed to the repo** (they are large, access-gated, and
licensed). Planned layout:

```
models/<task>/checkpoint.pt   # e.g. models/fiducials/checkpoint.pt
```

- Each task (`fiducials`, `marking`, `part-pose`, …) gets its own checkpoint
  and its own registry entry (endpoint port or in-process module).
- The Java side keeps a task registry (task → endpoint/port/timeout). The
  `LocalBottomVisionClient` already takes a port + timeout; per-task selection
  is a registry lookup (not implemented yet).
- The Python service lazy-loads the checkpoint for the requested task on first
  use (avoids loading several ViT backbones at startup).
- The wire contract stays identical across tasks — only the checkpoint and the
  model size change.

### 3.4 Training / fine-tuning (learning use)

- **Fiducial detection** is the default task: ViT-S/16 (~21M params, ~84 MB fp32
  / ~42 MB fp16) is the sensible starting point. ViT-S+/16 (~29M) if accuracy
  demands it. ViT-B/16 (~86M, ~344 MB fp32) is the heavy tier — only if needed.
- Frozen-backbone, head-only fine-tune: the trainable head is only a few MB, so
  per-task checkpoints are small (the base backbone can be shared; only the head
  is task-specific). This is the intended per-task weight story.
- Training data: bottom-vision frames (PNG/JPEG) + pose labels per task.

### 3.5 Licensing (private, non-commercial learning use)

- Meta's **DINOv3 weights are access-gated** (agreement via ai.meta.com). Do not
  commit, redistribute, or upload checkpoints anywhere public.
- Derivatives of DINOv3 (fine-tuned heads, per-task checkpoints) are covered by
  the **Meta DINOv3 License** — read it before sharing anything.
- This project is private, non-commercial, learning-use only. Do not ship,
  publish, or distribute the checkpoints or the fine-tuned weights.
- `samples/` and the repo must never contain checkpoints. `models/` (gitignored)
  holds local checkpoints only.

### 3.6 Intermediate steps (not done yet)

1. Python service skeleton (FastAPI/Flask) implementing the contract:
   `POST /v1/bottom-vision` → load task checkpoint → run inference → JSON out.
2. Task registry in Java: `Task → (endpoint, timeout)`; a `BottomVisionEngine`
   factory that selects the client by task.
3. Wire the engine into the bottom-vision path (`ReferenceBottomVision` or a
   new `AlternativeBottomVision`) behind a config flag, with the OpenCV pipeline
   as fallback.
4. Smoke-test against captured frames; add integration tests.

---

## 4. Build & packaging

### 4.1 macOS `.app` — done, verified

Toolchain: Temurin JDK 21 (`jpackage` 21.0.6), Maven 3.10.2 (brew), Java 11
target, checkstyle enforced.

```bash
# from the repo root
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home
mvn -q -B package -DskipTests

mkdir -p staging
cp target/openpnp-gui-0.0.1-alpha-SNAPSHOT.jar staging/
cp -r target/lib staging/lib

jpackage --type app-image --name OpenPnP --app-version 2.6.0 \
  --description "OpenPnP - Open Source SMT Pick and Place" \
  --vendor "OpenPnP" \
  --input staging \
  --main-jar openpnp-gui-0.0.1-alpha-SNAPSHOT.jar \
  --main-class org.openpnp.Main \
  --dest dist \
  --icon src/main/resources/icons/OpenPnP-logo.icns \
  --java-options "--add-opens=java.base/java.lang=ALL-UNNAMED" \
  --java-options "--add-opens=java.desktop/java.awt=ALL-UNNAMED" \
  --java-options "--add-opens=java.desktop/java.awt.color=ALL-UNNAMED"
```

Verified (2026-11-25):

- `dist/OpenPnP.app` — 356 MB (190 MB app + 165 MB bundled runtime).
- `./dist/OpenPnP.app/Contents/MacOS/OpenPnP --version` prints
  `2.6_2026-11-25_12-07-26.d5eb7d1` and exits 0 (safe headless smoke test;
  `--version` exits before any GUI is created).
- Note: `Main.getVersionString()` reads `VERSION.txt` from the working
  directory; a packaged app should be launched with a `VERSION.txt` next to the
  app or the version will show as `n/a`. `getVersionString()` fallback: `n/a`.

Gotchas:

- **Do not** use brew's `openjdk@27` as the build JDK — the system default and
  the packaging runtime should stay Temurin 21.
- `staging/` and `dist/` are build artifacts (not committed). `target/` and
  `installers/` are already gitignored.
- `--type app-image` produces an `.app` bundle, not a `.dmg`. A `.dmg` can be
  added later with `--type dmg` (unsigned, for private use).

### 4.2 Windows `.exe` — verified on a Windows host (2026-11-25)

- **jpackage cannot cross-build**: Windows artifacts must be built on Windows
  (local Windows PC or `windows-latest` runner). There is no macOS→Windows
  path. A local Windows 11 PC was used.
- `.exe` installers additionally need **WiX 3.x**. WiX 5/6 break jpackage.
  If WiX is unavailable, `--type app-image` still works and produces a Windows
  app directory (with an `.exe` launcher inside). WiX 3.14.1 was installed
  manually (`wix314-binaries.zip` extracted to a local `wix/` directory);
  `choco` is not present on the PC.
- `install4j` (the upstream path) is **not** available here: it is commercial,
  needs a license key + secrets, and upstream CI only deploys from
  `openpnp/openpnp` main/test.
- Verified-free path for the Windows side:

  ```bat
  mvn -q -B package -DskipTests
  jpackage --type exe --name OpenPnP --app-version 2.6.0 ^
    --input staging --main-jar openpnp-gui-0.0.1-alpha-SNAPSHOT.jar ^
    --main-class org.openpnp.Main --dest dist ^
    --icon src/main/resources/icons/OpenPnP.ico ^
    --java-options "--add-opens=java.base/java.lang=ALL-UNNAMED" ^
    --java-options "--add-opens=java.desktop/java.awt=ALL-UNNAMED" ^
    --java-options "--add-opens=java.desktop/java.awt.color=ALL-UNNAMED"
  ```

  Verified: `dist/OpenPnP-2.6.0.exe` (~236 MB) built and smoke-tested (app-image
  launcher `--version` exits 0, prints `2.6_2026-11-25_15-01-05.7da4a57`).
  The repo ships `src/main/resources/icons/OpenPnP.ico` (generated from the
  iconset, multi-size 16–256 px) for the Windows build; `.icns` remains the
  macOS icon.

### 4.3 Release pipeline (draft, partially validated)

`.github/workflows/release.yml` (new file; **not run on GitHub yet** — it only
runs on a GitHub push of a release; the individual build steps were re-verified
locally on macOS and Windows 11):

- Trigger: `release: published` + `workflow_dispatch`.
- **macOS job** (`macos-latest`): Temurin JDK 21 → `mvn -q -B package -DskipTests`
  → jpackage `--type app-image` → smoke test (`--version`) → zip `OpenPnP.app` →
  upload artifact + attach `OpenPnP.app.zip` to the GitHub release.
- **Windows job** (`windows-latest`): Temurin JDK 21 → `mvn -q -B package
  -DskipTests` → install WiX 3 → jpackage `--type exe` (with `--icon
  src/main/resources/icons/OpenPnP.ico`) → upload artifact +
  attach `.exe` to the GitHub release. Mirrors the locally-verified Windows
  build.
- Both jobs use `VERSION.txt` for `--app-version`.
- **Release attachments**: on `release: published`, both assets are attached to
  the GitHub release page via `softprops/action-gh-release`
  (`permissions: contents: write`; only runs for release events, not for manual
  `workflow_dispatch` runs). Release assets need single files — hence the `.app`
  is zipped first. `.app` zips are ~150 MB (GitHub's 2 GB asset limit is fine).
- **Linux is out of scope** (no workflow; upstream also only ships macOS).
- **Not run on GitHub yet**: the workflow is a draft to be validated on the
  first real release (both build steps were re-verified locally on macOS and
  Windows 11; the release-attachment step is untested).
- install4j (upstream) remains the production path for the original project;
  this pipeline is for the fork's private/learning builds.

### 4.4 Build scripts (this fork)

Two separate scripts — one per OS — both using paths relative to the repo (no
personal paths; tools are discovered via env vars or the user profile):

- `scripts/build-mac.sh` — macOS `.app` build (local, jpackage). `JAVA_HOME`
  is auto-discovered under `/Library/Java/JavaVirtualMachines` if unset.
  Steps: `mvn -q -B package -DskipTests` → staging → jpackage (`.app`) →
  `--version` smoke test.
- `scripts/build-windows.ps1` — Windows `.exe` build (run on a Windows host).
  Discovery via `JAVA_HOME` / `MAVEN_HOME` / `WIX_HOME` env vars, then the
  user profile (e.g. `%USERPROFILE%\maven`, `%USERPROFILE%\wix`), then
  `PATH`. Steps: `git pull --ff-only` → `mvn -q -B package -DskipTests` →
  staging → jpackage `.exe` (icon) → `--version` smoke test on a built
  app-image.

Both produce unsigned artifacts for private/learning use. `staging/`, `dist/`,
`dist-appimage/` and `models/` are build artifacts (gitignored).

---

## 5. Current state / checklist

- [x] Fork cloned; OpenCV pipeline verified in-process (not HTTP/port).
- [x] macOS `.app` built with jpackage and smoke-tested (`--version` exits 0).
- [x] Windows `.exe` built and smoke-tested (Windows 11 host, 2026-11-25).
- [ ] Release workflow validated on first release.
- [ ] DINOv3 service skeleton (Option A, §3).
- [ ] Per-task checkpoint registry + gitignore for `models/`.
- [ ] Engine wired into the bottom-vision path behind a config flag.

## 6. Open questions

- Should the packaged app ship a bundled Python DINOv3 service, or be launched
  as a separate process (service on 127.0.0.1, current contract)? Current plan:
  separate process; packaging a Python runtime is out of scope for now.
- Is a signed/notarized `.app` needed? Not for private learning use; the
  unsigned `.app` runs (Gatekeeper will warn on first launch).
- Should `.exe` be an installer (WiX) or an unpackaged app-image (no WiX)?
  Answer: installer (`.exe`) — verified 2026-11-25; app-image is used only for
  the headless smoke test.
