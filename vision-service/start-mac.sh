#!/bin/sh
# Start the OpenPnP vision service on macOS (arm64).
# Usage: ./start-mac.sh [models-dir] [extra --provider / --port / --host args]
# Requires python3 with onnxruntime + Pillow + numpy installed.
set -eu
cd "$(dirname "$0")"

MODELS_DIR="models/synthetic"
if [ $# -gt 0 ]; then
    case "$1" in
        --*) ;;
        *) MODELS_DIR="$1"; shift ;;
    esac
fi

exec python3 -m vision_service --models-dir "$MODELS_DIR" --host 127.0.0.1 --port "${PORT:-8010}" "$@"
