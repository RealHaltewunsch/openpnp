"""Encoder session management.

The encoder is loaded once per process. Task heads are loaded separately and
share this session; switching tasks never reloads the encoder.
"""

from __future__ import annotations

import logging
import platform
import uuid

import numpy as np
import onnxruntime as ort
from PIL import Image

log = logging.getLogger(__name__)

# Platform-specific accelerators, tried in preference order before CPU.
_PREFERRED_PROVIDERS = {
    "Darwin": ["CoreMLExecutionProvider"],
    "Windows": ["DMLExecutionProvider"],  # Windows-only DirectML provider
}


class EncoderLoadError(RuntimeError):
    """Raised when the encoder cannot be loaded or is incompatible."""


def detect_provider():
    """Pick the best available provider for the current platform.

    Preference order: platform-specific accelerator (CoreML on macOS arm64,
    DirectML on Windows x64), then CPU. Returns (provider, note).
    """
    available = ort.get_available_providers()
    log.info("Available providers: %s", available)
    system = platform.system()
    for provider in _PREFERRED_PROVIDERS.get(system, []):
        if provider in available:
            return provider, "platform accelerator"
    return "CPUExecutionProvider", "CPU fallback (no accelerator available)"


class Encoder:
    """A loaded encoder session, shared by every task head."""

    def __init__(self, spec, providers=None, session_id=None):
        self.spec = spec
        self.session_id = session_id or str(uuid.uuid4())

        if providers:
            self.requested_provider, note = providers[0], "forced"
        else:
            self.requested_provider, note = detect_provider()
        self.provider_note = note

        if not spec.path.is_file():
            raise EncoderLoadError(f"Encoder model not found: {spec.path}")

        preference = [self.requested_provider]
        if "CPUExecutionProvider" not in preference:
            preference.append("CPUExecutionProvider")
        self.preference = preference
        try:
            self.session = ort.InferenceSession(str(spec.path), providers=preference)
        except Exception as e:
            raise EncoderLoadError(f"Encoder cannot be loaded: {e}") from e

        # onnxruntime silently falls back through the preference list; log
        # the provider that actually ran.
        self.active_providers = list(self.session.get_providers())
        if self.active_providers and self.active_providers[0] != self.requested_provider:
            log.warning("Requested provider %s not active (got %s); running on %s",
                        self.requested_provider, self.active_providers, self.active_providers[0])
        log.info("Encoder loaded: version=%s provider=%s (%s) id=%s",
                 spec.version, self.active_providers[0], self.provider_note, self.session_id)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize, normalize and arrange the image exactly as the encoder expects."""
        prep = self.spec.preprocessing
        img = image.convert("RGB").resize((prep.resize[1], prep.resize[0]), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array(prep.mean, dtype=np.float32)
        std = np.array(prep.std, dtype=np.float32)
        arr = (arr - mean) / std
        if prep.channel_first:
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        arr = arr.reshape(1, *arr.shape)
        return arr

    def infer(self, image: Image.Image):
        """Run the encoder and return the feature vector."""
        tensor = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        features = self.session.run(None, {input_name: tensor})[0]
        return features

    def feature_dim(self):
        out = self.session.get_outputs()[0]
        shape = out.shape
        try:
            return int(shape[-1])
        except Exception:
            raise EncoderLoadError(f"Encoder output shape is not static: {shape}")
