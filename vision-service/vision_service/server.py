"""Loopback HTTP server for the vision service.

Routes:
  GET  /v1/health — liveness, provider and per-head status
  POST /v1/infer  — task-selected inference (JSON body)
"""

from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

log = logging.getLogger(__name__)


def _json_bytes(obj):
    return json.dumps(obj).encode("utf-8")


class VisionServiceHandler(BaseHTTPRequestHandler):
    server_version = "openpnp-vision/0.1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.info("%s %s", self.command, self.path)

    # ---- helpers ----

    def _send_json(self, status, obj):
        body = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, code, detail):
        self._send_json(status, {"status": "ERROR", "error": {"code": code, "detail": detail}})

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        try:
            n = int(length)
        except ValueError:
            return b""
        if n > 32 * 1024 * 1024:
            raise ValueError("Request body too large")
        return self.rfile.read(n)

    @property
    def service(self):
        return self.server.service

    # ---- routes ----

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/health":
            try:
                self._send_json(200, self.service.health())
            except Exception as e:
                log.exception("Health failed")
                self._error(500, "INTERNAL", str(e))
            return
        self._error(404, "NOT_FOUND", f"Unknown route: {path}")

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/v1/infer":
            self._error(404, "NOT_FOUND", f"Unknown route: {path}")
            return

        started = time.perf_counter()
        try:
            body = self._read_body()
        except ValueError as e:
            self._error(400, "INVALID_REQUEST", str(e))
            return

        try:
            result = self.service.infer_from_json_body(body)
            self._send_json(200, result)
        except Exception as e:
            code = getattr(e, "code", None) or "INFERENCE_ERROR"
            status_code = getattr(e, "status_code", None) or 500
            if code == "NOT_FOUND":
                status_code = 404
            self._error(status_code, code, str(e))

        elapsed = (time.perf_counter() - started) * 1000
        log.info("%s %s %.1fms", self.command, path, elapsed)


def create_server(host, port, service):
    server = ThreadingHTTPServer((host, port), VisionServiceHandler)
    server.service = service
    return server
