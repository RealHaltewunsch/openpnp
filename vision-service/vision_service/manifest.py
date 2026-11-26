"""Model manifest loading and validation.

The manifest binds a head version to an encoder version, feature layout and
preprocessing. It is the single source of truth for "is this head compatible
with the loaded encoder".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_CONTRACT_VERSIONS = {1}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: list
    dtype: str = "float32"


@dataclass(frozen=True)
class Preprocessing:
    resize: tuple  # (H, W)
    mean: tuple
    std: tuple
    layout: str = "CHW"
    channel_first: bool = True


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    version: str
    path: Path
    input: TensorSpec
    output: TensorSpec
    preprocessing: Preprocessing


@dataclass(frozen=True)
class HeadSpec:
    name: str
    version: str
    path: Path
    encoder_version: str
    input: TensorSpec
    output: TensorSpec
    output_layout: tuple

    @property
    def model_version(self):
        # Combined encoder+head version, as reported in inference responses.
        return self.encoder_version + "+" + self.version


@dataclass(frozen=True)
class Manifest:
    version: int
    contract_version: int
    encoder: EncoderSpec
    heads: dict = field(default_factory=dict)  # task name -> HeadSpec

    @property
    def tasks(self):
        return sorted(self.heads.keys())


class ManifestError(ValueError):
    """Raised when the manifest is missing, malformed or incompatible."""


def _tensor_spec(name, raw, where):
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: expected an object")
    if "name" not in raw:
        raise ManifestError(f"{where}: missing 'name'")
    shape = raw.get("shape")
    if not isinstance(shape, list) or not shape or not all(isinstance(d, int) and d > 0 for d in shape):
        raise ManifestError(f"{where}: 'shape' must be a non-empty list of positive ints")
    return TensorSpec(name=raw["name"], shape=list(shape), dtype=raw.get("dtype", "float32"))


def _preprocessing(raw, where):
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: expected an object")
    resize = raw.get("resize")
    if not (isinstance(resize, list) and len(resize) == 2 and all(isinstance(d, int) and d > 0 for d in resize)):
        raise ManifestError(f"{where}: 'resize' must be [H, W] of positive ints")
    mean = raw.get("mean", [0.0, 0.0, 0.0])
    std = raw.get("std", [1.0, 1.0, 1.0])
    if len(mean) != 3 or len(std) != 3:
        raise ManifestError(f"{where}: 'mean' and 'std' must be 3-element lists")
    if any(s == 0 for s in std):
        raise ManifestError(f"{where}: 'std' must not contain zeros")
    return Preprocessing(
        resize=(int(resize[0]), int(resize[1])),
        mean=tuple(float(m) for m in mean),
        std=tuple(float(s) for s in std),
        layout=raw.get("layout", "CHW"),
        channel_first=bool(raw.get("channel_first", True)),
    )


def load_manifest(path):
    """Load and validate the manifest at ``path``.

    Raises ManifestError on any structural problem. Missing encoder/head
    *files* are reported per-head/encoder status, not by this function, so
    that the service can start and report the problem through /v1/health.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError(f"Manifest is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ManifestError("Manifest must be a JSON object")

    contract_version = raw.get("contract_version")
    if contract_version not in SUPPORTED_CONTRACT_VERSIONS:
        raise ManifestError(
            f"Unsupported contract_version {contract_version!r} (supported: {sorted(SUPPORTED_CONTRACT_VERSIONS)})"
        )

    base = path.parent

    encoder_raw = raw.get("encoder")
    if not isinstance(encoder_raw, dict):
        raise ManifestError("Manifest is missing the 'encoder' object")
    enc_path = encoder_raw.get("path")
    if not isinstance(enc_path, str) or not enc_path:
        raise ManifestError("Encoder is missing 'path'")
    encoder = EncoderSpec(
        name=encoder_raw.get("name", "encoder"),
        version=encoder_raw.get("version", ""),
        path=(base / enc_path).resolve(),
        input=_tensor_spec("encoder.input", encoder_raw.get("input"), "encoder.input"),
        output=_tensor_spec("encoder.output", encoder_raw.get("output"), "encoder.output"),
        preprocessing=_preprocessing(encoder_raw.get("preprocessing"), "encoder.preprocessing"),
    )
    if not encoder.version:
        raise ManifestError("Encoder is missing 'version'")

    heads = {}
    heads_raw = raw.get("heads")
    if not isinstance(heads_raw, dict) or not heads_raw:
        raise ManifestError("Manifest is missing the 'heads' object")
    for task, head_raw in heads_raw.items():
        if not isinstance(head_raw, dict):
            raise ManifestError(f"Head '{task}': expected an object")
        head_path = head_raw.get("path")
        if not isinstance(head_path, str) or not head_path:
            raise ManifestError(f"Head '{task}': missing 'path'")
        layout = head_raw.get("output_layout")
        if not isinstance(layout, list) or not all(isinstance(f, str) for f in layout):
            raise ManifestError(f"Head '{task}': 'output_layout' must be a list of field names")
        if len(layout) != int(head_raw.get("output", {}).get("shape", [0])[-1] if isinstance(head_raw.get("output"), dict) else 0):
            raise ManifestError(f"Head '{task}': 'output_layout' length must match the output size")
        head = HeadSpec(
            name=task,
            version=head_raw.get("version", ""),
            path=(base / head_path).resolve(),
            encoder_version=head_raw.get("encoder_version", ""),
            input=_tensor_spec("input", head_raw.get("input"), f"head '{task}'.input"),
            output=_tensor_spec("output", head_raw.get("output"), f"head '{task}'.output"),
            output_layout=tuple(layout),
        )
        if not head.version:
            raise ManifestError(f"Head '{task}': missing 'version'")
        if not head.encoder_version:
            raise ManifestError(f"Head '{task}': missing 'encoder_version'")
        heads[task] = head

    return Manifest(
        version=int(raw.get("version", 1)),
        contract_version=contract_version,
        encoder=encoder,
        heads=heads,
    )
