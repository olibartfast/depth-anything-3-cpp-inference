#!/usr/bin/env python3
"""
Export a Depth Anything 3 monocular depth model to ONNX format.

The exported ONNX model accepts a single pre-processed image tensor with shape
    (1, 3, H, W)
and returns depth (and optionally confidence) maps.

Usage
-----
    python export_onnx.py \\
        --model-name da3-large \\
        --output model.onnx \\
        --process-res 504

    # Or load from local directory:
    python export_onnx.py \\
        --model-dir /path/to/local/model \\
        --output model.onnx

Requirements
------------
    pip install xformers torch>=2 torchvision
    pip install -e /path/to/Depth-Anything-3  # install the Python package

Notes
-----
- Supports DA3 monocular models (DA3Mono-Large, DA3Metric-Large, DA3-Small, DA3-Base, DA3-Large).
- The model input size is variable (dynamic axes). For fixed-size export, set --fixed-size.
- Output 0: depth  shape [1, H, W]
- Output 1: conf   shape [1, H, W]  (if the model produces it)
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Depth Anything 3 model to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-name",
        type=str,
        help="Hugging Face model name, e.g. 'da3-large', 'da3mono-large', 'da3metric-large'",
    )
    group.add_argument(
        "--model-dir",
        type=Path,
        help="Path to a local model directory (must contain config.json + model.safetensors)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.onnx"),
        help="Path for the exported ONNX file",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Longest-side resolution used to compute the input spatial dimensions",
    )
    parser.add_argument(
        "--fixed-size",
        action="store_true",
        help="Export with fixed H/W instead of dynamic axes (requires --process-res and a square image)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device on which to run the export trace",
    )
    return parser.parse_args()


def _patch_cartesian_prod_for_onnx() -> None:
    """Replace torch.cartesian_prod (2-arg) with an ONNX-traceable equivalent.

    Upstream RoPE (model/dinov2/layers/rope.py: PositionGetter) builds its
    patch grid via torch.cartesian_prod, which the TorchScript ONNX exporter
    cannot lower (UnsupportedOperatorError). The replacement uses only
    unsqueeze/expand/reshape/stack and is numerically identical:
    cartesian_prod(y, x)[k] == (y[i], x[j]) with i outer, j inner.
    Applied to torch globally but only affects this export process.
    """
    real_cartesian_prod = torch.cartesian_prod

    def _traceable_cartesian_prod(*tensors):
        if len(tensors) == 2:
            y, x = tensors
            yy = y.unsqueeze(1).expand(y.shape[0], x.shape[0]).reshape(-1)
            xx = x.unsqueeze(0).expand(y.shape[0], x.shape[0]).reshape(-1)
            return torch.stack([yy, xx], dim=1)
        return real_cartesian_prod(*tensors)

    torch.cartesian_prod = _traceable_cartesian_prod


class MonocularWrapper(torch.nn.Module):
    """
    Thin wrapper around DepthAnything3 that:
    - Accepts a plain (1, 3, H, W) image tensor (no batch-of-views dimension).
    - Returns only the depth and confidence maps as flat (1, H, W) tensors.

    This makes the ONNX graph much simpler than exporting the full API.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        # image: (1, 3, H, W)  →  add the N-views dimension → (1, 1, 3, H, W)
        imgs = image.unsqueeze(1).float()  # (1, 1, 3, H, W)
        # NOTE: 4th positional is export_feat_layers — must be [] (not None);
        # upstream defaults to [] and `i in None` raises TypeError even eagerly.
        output = self.model.model(imgs, None, None, [], False, False, "saddle_balanced")

        # depth: (B, N, 1, H, W) → squeeze batch+views+channel → (H, W) → add batch dim
        depth = output["depth"].squeeze(0).squeeze(0).squeeze(0)  # (H, W)
        depth = depth.unsqueeze(0)  # (1, H, W)

        conf = output.get("depth_conf", None)
        if conf is not None:
            conf = conf.squeeze(0).squeeze(0)  # (H, W)
            conf = conf.unsqueeze(0)  # (1, H, W)
            return depth, conf

        return (depth,)


def main() -> int:
    args = parse_args()

    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        print(
            "ERROR: Cannot import depth_anything_3. "
            "Install it with:  pip install -e /path/to/Depth-Anything-3",
            file=sys.stderr,
        )
        return 1

    device = torch.device(args.device)
    print(f"Loading model on {device}…")

    if args.model_dir is not None:
        model = DepthAnything3.from_pretrained(str(args.model_dir))
    else:
        model = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name.upper()}")

    model = model.to(device=device)
    model.eval()

    _patch_cartesian_prod_for_onnx()

    wrapper = MonocularWrapper(model)
    wrapper.eval()

    # Build a representative dummy input
    # Use a square image equal to process_res for simplicity;
    # the patch_size constraint (divisible by 14) is already satisfied since
    # compute_output_size(process_res, process_res, process_res, 14) == (process_res, process_res)
    # when process_res is already divisible by 14. We round it here.
    patch_size = 14
    res = (args.process_res // patch_size) * patch_size
    if res <= 0:
        res = patch_size

    dummy_input = torch.zeros(1, 3, res, res, dtype=torch.float32, device=device)

    print(f"Tracing with input shape: {list(dummy_input.shape)}")

    dynamic_axes: dict | None = None
    if not args.fixed_size:
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 1: "height", 2: "width"},
        }
        # Try to detect whether conf output exists
        with torch.no_grad():
            out = wrapper(dummy_input)
        if len(out) > 1:
            dynamic_axes["conf"] = {0: "batch", 1: "height", 2: "width"}

    output_names = ["depth"]
    with torch.no_grad():
        out = wrapper(dummy_input)
    if len(out) > 1:
        output_names.append("conf")

    print(f"Output names: {output_names}")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            opset_version=args.opset,
            input_names=["image"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,  # legacy tracer: dynamo export fails on DA3 (SymInt/control-flow)
        )

    print(f"Model exported to: {output_path}")

    # Quick validation
    try:
        import onnx
        model_onnx = onnx.load(str(output_path))
        onnx.checker.check_model(model_onnx)
        print("ONNX model validated successfully.")
    except ImportError:
        print("onnx package not installed — skipping validation.")
    except Exception as exc:  # noqa: BLE001
        print(f"ONNX validation warning: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
