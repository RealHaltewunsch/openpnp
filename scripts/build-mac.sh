#!/bin/bash
# macOS .app build (local, jpackage). Run from anywhere — paths are relative to the repo.
# Prereq: Temurin JDK 21 (JAVA_HOME, or auto-discovered under /Library/Java/JavaVirtualMachines)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${JAVA_HOME:-}" ]; then
  jdk_dir="$(ls -d /Library/Java/JavaVirtualMachines/temurin-21* 2>/dev/null | head -1)"
  if [ -z "$jdk_dir" ]; then
    echo "ERROR: JDK 21 not found. Set JAVA_HOME." >&2
    exit 1
  fi
  export JAVA_HOME="$jdk_dir/Contents/Home"
fi

echo "=== java ==="; java -version
echo "=== mvn package ==="
mvn -q -B package -DskipTests

echo "=== staging ==="
rm -rf staging dist
mkdir -p staging
cp target/openpnp-gui-0.0.1-alpha-SNAPSHOT.jar staging/
cp -r target/lib staging/lib

VERSION="$(cat VERSION.txt).0"
echo "=== jpackage (app-version: $VERSION) ==="
jpackage --type app-image --name OpenPnP --app-version "$VERSION" \
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

echo "=== smoke test ==="
./dist/OpenPnP.app/Contents/MacOS/OpenPnP --version
echo "=== done ==="
ls -la dist/OpenPnP.app
