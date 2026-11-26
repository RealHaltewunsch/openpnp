#!/usr/bin/env python3
"""Generate the synthetic encoder + two test heads + a fixture image.

This is a TEST-ONLY model package. It is not a trained DINOv3 model and does
not pretend to be one. It exists to prove the service plumbing: encoder
load-once, shared session, head loading, warmup, task switching, and
normalized-output denormalization — against the same interface trained models
will later use.

The synthetic encoder is a hand-crafted deterministic ONNX graph that
computes spatial statistics of the input image (mean RGB, luminance mean /
std / contrast, dark-region fraction / extent / center / mean luminance,
weight-based center of mass, min / max luminance) — 16 features. The heads
are linear ONNX graphs fit to known-good synthetic renders.

Usage:
    python3 scripts/generate_synthetic.py --out <models-dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RESIZE_H, RESIZE_W = 32, 32
FEATURE_DIM = 16
FIXTURE_SIZE = (320, 240)  # (W, H) of the saved fixture image
BATCH_SIZE = 64

# Encoder / head version strings. Change these when the models change.
ENCODER_VERSION = "synthetic-1.0.0"
HEAD_VERSION = "synthetic-1.0.0"


# ---------------------------------------------------------------------------
# Synthetic image rendering
# ---------------------------------------------------------------------------

def render_image(rng, rect_center_frac=None, rect_size_frac=None, angle_deg=None):
    """Render a synthetic IC bottom: dark rounded rect on a light background.

    Returns (PIL.Image, truth dict) with truth containing the rect's center,
    size (fractions) and angle (degrees).
    """
    from PIL import Image, ImageDraw

    W, H = FIXTURE_SIZE
    cx = rect_center_frac[0] if rect_center_frac else rng.uniform(0.3, 0.7)
    cy = rect_center_frac[1] if rect_center_frac else rng.uniform(0.3, 0.7)
    sw = rect_size_frac[0] if rect_size_frac else rng.uniform(0.35, 0.6)
    sh = rect_size_frac[1] if rect_size_frac else rng.uniform(0.3, 0.55)
    angle = angle_deg if angle_deg is not None else rng.uniform(-30, 30)

    img = Image.new("RGB", (W, H), (225, 225, 225))
    draw = ImageDraw.Draw(img)

    rw, rh = int(W * sw), int(H * sh)
    rect = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(rect)
    rdraw.rounded_rectangle([2, 2, rw - 3, rh - 3], radius=min(8, rw // 5), fill=(35, 35, 35, 255))

    rotated = rect.rotate(angle, resample=Image.BICUBIC, center=(rw / 2, rh / 2))
    px, py = int(W * cx - rw / 2), int(H * cy - rh / 2)
    img.paste(rotated, (px, py), rotated)

    truth = {
        "center_x": cx,
        "center_y": cy,
        "size_w": sw,
        "size_h": sh,
        "angle_deg": angle,
    }
    return img, truth


# ---------------------------------------------------------------------------
# ONNX graph construction
# ---------------------------------------------------------------------------

def _int_const(name, values):
    from onnx import helper, TensorProto

    return helper.make_node(
        "Constant", inputs=[], outputs=[name], name=name,
        value=helper.make_tensor(name, TensorProto.INT64, [len(values)], list(values)),
    )


def _axes_const(name, axes):
    from onnx import helper, TensorProto

    return helper.make_node(
        "Constant", inputs=[], outputs=[name], name=name,
        value=helper.make_tensor(name, TensorProto.INT64, [len(axes)], list(axes)),
    )


def _constant_scalar(name, value):
    import onnx
    from onnx import helper, TensorProto

    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=name,
        value=helper.make_tensor(name, TensorProto.FLOAT, [1, 1], [float(value)]),
    )


def _coordinate_grid(kind):
    """Return (1,1,H,W) float32 grid: x-kind -> x/W, y-kind -> y/H."""
    vals = np.zeros((1, 1, RESIZE_H, RESIZE_W), dtype=np.float32)
    for y in range(RESIZE_H):
        for x in range(RESIZE_W):
            vals[0, 0, y, x] = x / RESIZE_W if kind == "x" else y / RESIZE_H
    return vals


def build_encoder_model():
    """Hand-crafted synthetic encoder. Returns an onnx.ModelProto."""
    import onnx
    from onnx import helper, TensorProto

    H, W = RESIZE_H, RESIZE_W

    nodes = []

    # Constants
    nodes.append(_constant_scalar("c_threshold", 0.5))
    nodes.append(_constant_scalar("c_one", 1.0))
    nodes.append(_constant_scalar("c_eps", 1e-6))
    nodes.append(_constant_scalar("c_zero", 0.0))
    nodes.append(_constant_scalar("c_minus1", -1.0))
    nodes.append(_constant_scalar("c_1e9", 1e9))

    x_coord = _coordinate_grid("x")
    y_coord = _coordinate_grid("y")

    nodes.append(helper.make_node(
        "Constant", inputs=[], outputs=["x_coord"], name="x_coord",
        value=helper.make_tensor("x_coord", TensorProto.FLOAT, [1, 1, H, W], x_coord.ravel().tolist()),
    ))
    nodes.append(helper.make_node(
        "Constant", inputs=[], outputs=["y_coord"], name="y_coord",
        value=helper.make_tensor("y_coord", TensorProto.FLOAT, [1, 1, H, W], y_coord.ravel().tolist()),
    ))

    # Scalar constants for luminance weights (as (1,1) tensors)
    for cname, cval in [("c_r", 0.299), ("c_g", 0.587), ("c_b", 0.114)]:
        nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[cname], name=cname,
            value=helper.make_tensor(cname, TensorProto.FLOAT, [1, 1], [cval]),
        ))

    # Per-channel slices: r,g,b each (1,1,H,W). The Slice schema in this
    # onnxruntime takes starts/ends/axes as input tensors.
    nodes.append(_int_const("slice_axes", [1]))
    nodes.append(_axes_const("axes_23", [2, 3]))
    for i, cname in enumerate(["r", "g", "b"]):
        start_name = f"slice_start_{i}"
        end_name = f"slice_end_{i}"
        nodes.append(_int_const(start_name, [i]))
        nodes.append(_int_const(end_name, [i + 1]))
        nodes.append(helper.make_node(
            "Slice",
            inputs=["image", start_name, end_name, "slice_axes"],
            outputs=[cname],
            name=cname,
        ))

    # Luminance: 0.299*r + 0.587*g + 0.114*b
    nodes.append(helper.make_node("Mul", ["r", "c_r"], ["r_w"], name="r_w"))
    nodes.append(helper.make_node("Mul", ["g", "c_g"], ["g_w"], name="g_w"))
    nodes.append(helper.make_node("Mul", ["b", "c_b"], ["b_w"], name="b_w"))
    nodes.append(helper.make_node("Add", ["r_w", "g_w"], ["lum_rg"], name="lum_rg"))
    nodes.append(helper.make_node("Add", ["lum_rg", "b_w"], ["lum"], name="lum"))

    # Mean RGB per channel: reduce over axes 2,3 -> (1,1)
    for cname, out in [("r", "mean_r"), ("g", "mean_g"), ("b", "mean_b")]:
        nodes.append(helper.make_node(
            "ReduceMean", [cname, "axes_23"], [out], name=out, keepdims=0,
        ))

    # Luminance statistics
    nodes.append(helper.make_node(
        "ReduceMean", ["lum", "axes_23"], ["lum_mean"], name="lum_mean", keepdims=0,
    ))
    nodes.append(helper.make_node(
        "ReduceMax", ["lum", "axes_23"], ["lum_max"], name="lum_max", keepdims=0,
    ))
    nodes.append(helper.make_node(
        "ReduceMin", ["lum", "axes_23"], ["lum_min"], name="lum_min", keepdims=0,
    ))
    nodes.append(helper.make_node("Sub", ["lum_max", "lum_min"], ["contrast"], name="contrast"))

    # Luminance std: (lum - mean)^2, mean, sqrt
    nodes.append(helper.make_node(
        "ReduceMean", ["lum", "axes_23"], ["lum_mean_keep"], name="lum_mean_keep", keepdims=1,
    ))
    nodes.append(helper.make_node("Sub", ["lum", "lum_mean_keep"], ["lum_diff"], name="lum_diff"))
    nodes.append(helper.make_node("Mul", ["lum_diff", "lum_diff"], ["lum_sq"], name="lum_sq"))
    nodes.append(helper.make_node(
        "ReduceMean", ["lum_sq", "axes_23"], ["lum_mean_sq"], name="lum_mean_sq", keepdims=0,
    ))
    nodes.append(helper.make_node("Sqrt", ["lum_mean_sq"], ["lum_std"], name="lum_std"))

    # Dark region (luminance < 0.5)
    nodes.append(helper.make_node("Less", ["lum", "c_threshold"], ["dark_bool"], name="dark_bool"))
    nodes.append(helper.make_node(
        "Cast", ["dark_bool"], ["dark_float"], name="dark_float", to=TensorProto.FLOAT,
    ))
    nodes.append(helper.make_node(
        "ReduceMean", ["dark_float", "axes_23"], ["dark_frac"], name="dark_frac", keepdims=0,
    ))
    nodes.append(helper.make_node(
        "ReduceSum", ["dark_float", "axes_23"], ["sum_dark"], name="sum_dark", keepdims=0,
    ))
    nodes.append(helper.make_node("Add", ["sum_dark", "c_eps"], ["sum_dark_eps"], name="sum_dark_eps"))

    # Dark extents
    nodes.append(helper.make_node(
        "Where", ["dark_bool", "x_coord", "c_minus1"], ["x_masked_max"], name="x_masked_max",
    ))
    nodes.append(helper.make_node(
        "ReduceMax", ["x_masked_max", "axes_23"], ["x_max"], name="x_max", keepdims=0,
    ))
    nodes.append(helper.make_node(
        "Where", ["dark_bool", "x_coord", "c_1e9"], ["x_masked_min"], name="x_masked_min",
    ))
    nodes.append(helper.make_node(
        "ReduceMin", ["x_masked_min", "axes_23"], ["x_min"], name="x_min", keepdims=0,
    ))
    nodes.append(helper.make_node("Sub", ["x_max", "x_min"], ["extent_x_raw"], name="extent_x_raw"))
    nodes.append(helper.make_node("Less", ["extent_x_raw", "c_zero"], ["extent_x_neg"], name="extent_x_neg"))
    nodes.append(helper.make_node(
        "Where", ["extent_x_neg", "c_zero", "extent_x_raw"], ["extent_x"], name="extent_x",
    ))

    nodes.append(helper.make_node(
        "Where", ["dark_bool", "y_coord", "c_minus1"], ["y_masked_max"], name="y_masked_max",
    ))
    nodes.append(helper.make_node(
        "ReduceMax", ["y_masked_max", "axes_23"], ["y_max"], name="y_max", keepdims=0,
    ))
    nodes.append(helper.make_node(
        "Where", ["dark_bool", "y_coord", "c_1e9"], ["y_masked_min"], name="y_masked_min",
    ))
    nodes.append(helper.make_node(
        "ReduceMin", ["y_masked_min", "axes_23"], ["y_min"], name="y_min", keepdims=0,
    ))
    nodes.append(helper.make_node("Sub", ["y_max", "y_min"], ["extent_y_raw"], name="extent_y_raw"))
    nodes.append(helper.make_node("Less", ["extent_y_raw", "c_zero"], ["extent_y_neg"], name="extent_y_neg"))
    nodes.append(helper.make_node(
        "Where", ["extent_y_neg", "c_zero", "extent_y_raw"], ["extent_y"], name="extent_y",
    ))

    # Dark center-of-mass (fraction coords)
    nodes.append(helper.make_node("Mul", ["x_coord", "dark_float"], ["x_dark"], name="x_dark"))
    nodes.append(helper.make_node(
        "ReduceSum", ["x_dark", "axes_23"], ["sum_x_dark"], name="sum_x_dark", keepdims=0,
    ))
    nodes.append(helper.make_node("Div", ["sum_x_dark", "sum_dark_eps"], ["dark_center_x"], name="dark_center_x"))

    nodes.append(helper.make_node("Mul", ["y_coord", "dark_float"], ["y_dark"], name="y_dark"))
    nodes.append(helper.make_node(
        "ReduceSum", ["y_dark", "axes_23"], ["sum_y_dark"], name="sum_y_dark", keepdims=0,
    ))
    nodes.append(helper.make_node("Div", ["sum_y_dark", "sum_dark_eps"], ["dark_center_y"], name="dark_center_y"))

    # Dark-region mean luminance
    nodes.append(helper.make_node("Mul", ["lum", "dark_float"], ["lum_dark"], name="lum_dark"))
    nodes.append(helper.make_node(
        "ReduceSum", ["lum_dark", "axes_23"], ["sum_lum_dark"], name="sum_lum_dark", keepdims=0,
    ))
    nodes.append(helper.make_node("Div", ["sum_lum_dark", "sum_dark_eps"], ["dark_lum_mean"], name="dark_lum_mean"))

    # Weight-based center of mass (weight = 1 - lum)
    nodes.append(helper.make_node("Sub", ["c_one", "lum"], ["weight"], name="weight"))
    nodes.append(helper.make_node("Mul", ["x_coord", "weight"], ["x_w"], name="x_w"))
    nodes.append(helper.make_node(
        "ReduceSum", ["x_w", "axes_23"], ["sum_x_w"], name="sum_x_w", keepdims=0,
    ))
    nodes.append(helper.make_node("Mul", ["y_coord", "weight"], ["y_w"], name="y_w"))
    nodes.append(helper.make_node(
        "ReduceSum", ["y_w", "axes_23"], ["sum_y_w"], name="sum_y_w", keepdims=0,
    ))
    nodes.append(helper.make_node(
        "ReduceSum", ["weight", "axes_23"], ["sum_w"], name="sum_w", keepdims=0,
    ))
    nodes.append(helper.make_node("Add", ["sum_w", "c_eps"], ["sum_w_eps"], name="sum_w_eps"))
    nodes.append(helper.make_node("Div", ["sum_x_w", "sum_w_eps"], ["weight_center_x"], name="weight_center_x"))
    nodes.append(helper.make_node("Div", ["sum_y_w", "sum_w_eps"], ["weight_center_y"], name="weight_center_y"))

    # Feature vector
    features = [
        "mean_r", "mean_g", "mean_b",
        "lum_mean", "lum_std", "contrast",
        "dark_frac", "extent_x", "extent_y",
        "dark_center_x", "dark_center_y", "dark_lum_mean",
        "weight_center_x", "weight_center_y",
        "lum_max", "lum_min",
    ]
    nodes.append(helper.make_node(
        "Concat", features, ["features_concat"], name="features_concat", axis=0,
    ))
    # (16,1) -> (1,16)
    nodes.append(helper.make_node(
        "Constant", inputs=[], outputs=["shape_out"], name="shape_out",
        value=helper.make_tensor("shape_out", TensorProto.INT64, [2], [1, FEATURE_DIM]),
    ))
    nodes.append(helper.make_node(
        "Reshape", ["features_concat", "shape_out"], ["features"], name="features",
    ))

    image_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, H, W])
    features_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, FEATURE_DIM])
    graph = helper.make_graph(nodes, "synthetic-encoder", [image_info], [features_info])

    model = helper.make_model(graph, producer_name="openpnp-vision-service")
    model.ir_version = 10
    # Keep the graph self-contained; explicit opset.
    return model


# ---------------------------------------------------------------------------
# Head construction
# ---------------------------------------------------------------------------

def build_head_model(task, weights, biases):
    """Linear ONNX head. weights: (out, 16) float32, biases: (out,) float32.

    Output semantics per task:
      bottom-vision: center_x, center_y, size_w, size_h, angle_deg, confidence
        center/size/confidence -> Sigmoid; angle_deg -> 180 * tanh
      homing: center_x, center_y, visibility, confidence -> Sigmoid
    """
    import onnx
    from onnx import helper, TensorProto

    layout = HEAD_LAYOUTS[task]
    out_dim = len(layout)

    nodes = []
    for i in range(out_dim):
        nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[f"w_{i}"], name=f"w_{i}",
            value=helper.make_tensor(f"w_{i}", TensorProto.FLOAT, [FEATURE_DIM, 1],
                                     weights[i].reshape(FEATURE_DIM, 1).ravel().tolist()),
        ))
        nodes.append(helper.make_node(
            "Constant", inputs=[], outputs=[f"b_{i}"], name=f"b_{i}",
            value=helper.make_tensor(f"b_{i}", TensorProto.FLOAT, [1, 1], [float(biases[i])]),
        ))
        nodes.append(helper.make_node(
            "Gemm", ["features", f"w_{i}", f"b_{i}"], [f"raw_{i}"], name=f"raw_{i}",
        ))
        if layout[i] == "angle_deg":
            nodes.append(helper.make_node("Tanh", [f"raw_{i}"], [f"act_{i}"], name=f"act_{i}"))
            nodes.append(helper.make_node(
                "Constant", inputs=[], outputs=[f"c180_{i}"], name=f"c180_{i}",
                value=helper.make_tensor(f"c180_{i}", TensorProto.FLOAT, [1, 1], [180.0]),
            ))
            nodes.append(helper.make_node("Mul", [f"act_{i}", f"c180_{i}"], [f"out_{i}"], name=f"out_{i}"))
        else:
            nodes.append(helper.make_node("Sigmoid", [f"raw_{i}"], [f"out_{i}"], name=f"out_{i}"))

    nodes.append(helper.make_node(
        "Concat", [f"out_{i}" for i in range(out_dim)], ["result"], name="result", axis=1,
    ))

    features_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, FEATURE_DIM])
    result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, out_dim])
    graph = helper.make_graph(nodes, f"head-{task}", [features_info], [result_info])

    model = helper.make_model(graph, producer_name="openpnp-vision-service")
    model.ir_version = 10
    return model


HEAD_LAYOUTS = {
    "bottom-vision": ["center_x", "center_y", "size_w", "size_h", "angle_deg", "confidence"],
    "homing": ["center_x", "center_y", "visibility", "confidence"],
}


def train_head(task, features, targets):
    """Fit a linear head to (features -> targets) via least squares on the
    inverse-sigmoid/tanh logit space.

    features: (N, 16) float32
    targets: dict with the same field names as HEAD_LAYOUTS[task]; values
             are the desired normalized outputs.
    Returns (weights (out, 16), biases (out,)).
    """
    layout = HEAD_LAYOUTS[task]
    N = len(features)
    out_dim = len(layout)

    # Build the regression target per field.
    regression = np.zeros((N, out_dim), dtype=np.float64)
    for j, name in enumerate(layout):
        raw = np.asarray(targets[name], dtype=np.float64)
        if name == "angle_deg":
            # angle in [-180, 180] -> tanh(raw) = angle / 180
            regression[:, j] = np.arctanh(np.clip(raw / 180.0, -0.99, 0.99))
        else:
            # sigmoid output in (0, 1)
            clipped = np.clip(raw, 1e-4, 1 - 1e-4)
            regression[:, j] = np.log(clipped / (1 - clipped))

    weights = np.zeros((out_dim, FEATURE_DIM), dtype=np.float64)
    biases = np.zeros(out_dim, dtype=np.float64)

    # Augmented least squares: [F | 1] [W^T | b]^T = regression
    A = np.column_stack([features, np.ones(N)])
    # Solve per output field independently.
    for j in range(out_dim):
        sol, _, _, _ = np.linalg.lstsq(A, regression[:, j], rcond=None)
        weights[j] = sol[:FEATURE_DIM]
        biases[j] = sol[FEATURE_DIM]

    return weights, biases


# ---------------------------------------------------------------------------
# Manifest / fixture writing
# ---------------------------------------------------------------------------

def write_manifest(models_dir, encoder_version=ENCODER_VERSION):
    manifest = {
        "version": 1,
        "contract_version": 1,
        "encoder": {
            "name": "synthetic-encoder",
            "version": encoder_version,
            "path": "encoder.onnx",
            "input": {"name": "image", "shape": [1, 3, RESIZE_H, RESIZE_W], "dtype": "float32"},
            "output": {"name": "features", "shape": [1, FEATURE_DIM], "dtype": "float32"},
            "preprocessing": {
                "resize": [RESIZE_H, RESIZE_W],
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "layout": "CHW",
                "channel_first": True,
            },
        },
        "heads": {
            task: {
                "version": HEAD_VERSION,
                "path": f"heads/{task}.onnx",
                "encoder_version": encoder_version,
                "input": {"name": "features", "shape": [1, FEATURE_DIM], "dtype": "float32"},
                "output": {"name": "result", "shape": [1, len(layout)], "dtype": "float32"},
                "output_layout": layout,
            }
            for task, layout in HEAD_LAYOUTS.items()
        },
    }
    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic test model package")
    parser.add_argument("--out", required=True, help="Output models directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models_dir = Path(args.out)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "heads").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # 1. Build the synthetic encoder
    import onnx
    import onnxruntime as ort

    encoder_model = build_encoder_model()
    encoder_path = models_dir / "encoder.onnx"
    onnx.save(encoder_model, str(encoder_path))
    encoder_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])

    # 2. Render batch + compute features + ground truth
    def preprocess(image):
        img = image.convert("RGB").resize((RESIZE_W, RESIZE_H), __import__("PIL.Image", fromlist=["Image"]).BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # CHW
        return arr.reshape(1, 3, RESIZE_H, RESIZE_W)

    batch = []
    targets_bottom = {"center_x": [], "center_y": [], "size_w": [], "size_h": [], "angle_deg": [], "confidence": []}
    targets_homing = {"center_x": [], "center_y": [], "visibility": [], "confidence": []}

    for i in range(args.batch_size):
        image, truth = render_image(rng)
        feats = encoder_session.run(None, {"image": preprocess(image)})[0].reshape(-1)
        batch.append(feats)

        targets_bottom["center_x"].append(truth["center_x"])
        targets_bottom["center_y"].append(truth["center_y"])
        targets_bottom["size_w"].append(truth["size_w"])
        targets_bottom["size_h"].append(truth["size_h"])
        targets_bottom["angle_deg"].append(truth["angle_deg"])
        targets_bottom["confidence"].append(0.9)

        targets_homing["center_x"].append(truth["center_x"])
        targets_homing["center_y"].append(truth["center_y"])
        targets_homing["visibility"].append(0.9)
        targets_homing["confidence"].append(0.9)

    features = np.stack(batch, axis=0).astype(np.float32)

    # 3. Train the heads
    for task, targets in [("bottom-vision", targets_bottom), ("homing", targets_homing)]:
        weights, biases = train_head(task, features, targets)
        head_model = build_head_model(task, weights, biases)
        head_path = models_dir / "heads" / f"{task}.onnx"
        onnx.save(head_model, str(head_path))
        print(f"trained head '{task}': weights shape {weights.shape}, biases {biases.shape}")

    # 4. Write the manifest
    write_manifest(models_dir)

    # 5. Save the fixture image (slightly rotated dark rounded rect)
    fixture_rng = np.random.default_rng(7)
    fixture_image, fixture_truth = render_image(
        fixture_rng,
        rect_center_frac=(0.5, 0.5),
        rect_size_frac=(0.55, 0.5),
        angle_deg=12.0,
    )
    fixture_path = models_dir / "fixture.png"
    fixture_image.save(fixture_path)
    fixture_meta = {
        "width": FIXTURE_SIZE[0],
        "height": FIXTURE_SIZE[1],
        "expected": {
            "center_px": [fixture_truth["center_x"] * FIXTURE_SIZE[0], fixture_truth["center_y"] * FIXTURE_SIZE[1]],
            "size_px": [fixture_truth["size_w"] * FIXTURE_SIZE[0], fixture_truth["size_h"] * FIXTURE_SIZE[1]],
            "angle_deg": fixture_truth["angle_deg"],
        },
    }
    (models_dir / "fixture.json").write_text(json.dumps(fixture_meta, indent=2) + "\n", encoding="utf-8")

    print(f"synthetic package written to {models_dir}")
    print(f"fixture: {fixture_path} ({fixture_image.width}x{fixture_image.height})")


if __name__ == "__main__":
    main()
