"""Entry point: python -m vision_service --models-dir <dir> [--port 8010] [--host 127.0.0.1] [--provider cpu|auto]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local vision service for OpenPnP")
    parser.add_argument("--models-dir", required=True,
                        help="Directory containing manifest.json and model files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--provider", default=None,
                        help="Force an ONNX Runtime provider (e.g. CPUExecutionProvider, CoreMLExecutionProvider). Default: auto")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from .service import VisionService
    from .server import create_server

    service = VisionService(Path(args.models_dir), provider=args.provider)
    if not service.encoder_ok:
        logging.getLogger(__name__).warning(
            "Service is degraded: %s. Starting anyway so /v1/health can report the problem.",
            service.manifest_error,
        )

    server = create_server(args.host, args.port, service)
    logging.getLogger(__name__).info(
        "vision-service listening on http://%s:%d (encoder=%s)",
        args.host, args.port,
        service.encoder.spec.version if service.encoder_ok else "unavailable",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
