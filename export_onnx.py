#!/usr/bin/env python3
"""
Export a Depth Anything 3 model to ONNX format.

Native DA3 tasks (--task) and the ONNX outputs each one produces:

| --task         | Input              | Outputs (best-effort; missing heads warn)        |
|----------------|--------------------|--------------------------------------------------|
| mono           | (1, 3, H, W)       | depth[, conf]                                    |
| metric         | (1, 3, H, W)       | depth[, conf] (meters via focal*net/300 in C++)  |
| multiview      | (N, 3, H, W)       | depth[, conf] with N-view batch dim              |
| pose           | (N, 3, H, W)       | depth[, conf], extrinsics [N,3,4], intrinsics [N,3,3] |
| reconstruction | (N, 3, H, W)       | depth[, conf], pointmap                          |
| gaussian       | (N, 3, H, W)       | depth[, conf], gaussian_params                   |

Usage
-----
    python export_onnx.py \\
        --model-name da3-large \\
        --output model.onnx \\
        --process-res 504

    python export_onnx.py \\
        --model-name da3-large-1.1 \\
        --output model_pose.onnx \\
        --task pose --use-ray-pose

    # Or load from local directory:
    python export_onnx.py \\
        --model-dir /path/to/local/model \\
        --output model.onnx


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
        help="Hugging Face model name, e.g. 'da3-large', 'da3mono-large', 'da3metric-large'. "
        "Camera-depth ids (d405, d435, l515, kinect, zed2i-neural, zed2i-quality, base) "
        "resolve under the depth-anything/camera-depth-models collection.",
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
        help="Export with fixed H/W instead of dynamic axes (requires --process-res and a square image). "
        "NOTE: dynamic-axes graphs are trace-shape-locked (finding F-1): the upstream RoPE tables specialize "
        "at trace time, so off-shape inputs (e.g. 252x504 from a 504 trace) fail in broadcast. Prefer "
        "--fixed-size for TensorRT/static-shape use.",
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
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use the dynamo ONNX exporter (static shapes) instead of the legacy "
        "tracer. REQUIRED for models whose fp32 weights exceed ~2GB (Giant, "
        "Nested): the legacy path cannot serialize protos past the 2GB protobuf "
        "limit and silently writes a weightless graph. The dynamo path saves "
        "weights as external data automatically.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mono",
        choices=["mono", "metric", "multiview", "pose", "reconstruction", "gaussian"],
        help="Native DA3 task to export (selects the ONNX output set; "
        "C++ --task must match). multiview/pose/reconstruction/gaussian use an "
        "N-view batch input (N, 3, H, W).",
    )
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        help="Derive camera pose from the ray head (slower, more accurate). "
        "Forwarded as use_ray_pose to model.model(); only affects pose-capable "
        "checkpoints (DA3-LARGE/GIANT, not METRIC).",
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


def _patch_rope_for_dynamo() -> None:
    """Make upstream 2D RoPE dynamo-exportable (static shapes).

    Upstream forward does `max_position = int(positions.max()) + 1`, which
    raises under torch.export (no int() of a FakeTensor). The frequency table
    is only used as an embedding lookup by position index, so sizing it by the
    token count (positions.shape[1] >= max index + 1 for any full patch grid)
    is exactly equivalent. Applied only for --dynamo exports.
    """
    import torch.nn.functional as F

    from depth_anything_3.model.dinov2.layers import rope as rope_mod

    cls = rope_mod.RotaryPositionEmbedding2D
    orig_rotate = cls._rotate_features

    def _forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must be (B, N, 2)"
        feature_dim = tokens.size(-1) // 2
        # NOTE: differs from upstream only here (no int(tensor)).
        max_position = positions.shape[1]
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        cos = F.embedding(positions[..., 0], cos_comp)[:, None, :, :]
        sin = F.embedding(positions[..., 0], sin_comp)[:, None, :, :]
        vertical_features = (vertical_features * cos) + (orig_rotate(vertical_features) * sin)
        cos = F.embedding(positions[..., 1], cos_comp)[:, None, :, :]
        sin = F.embedding(positions[..., 1], sin_comp)[:, None, :, :]
        horizontal_features = (horizontal_features * cos) + (orig_rotate(horizontal_features) * sin)
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    cls.forward = _forward


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

        # depth: [1,1,H,W] (or legacy [B,N,1,H,W]) → squeeze all size-1 dims → [H,W].
        # NOTE: bare squeeze() (not chained squeeze(0)) — the dynamo exporter
        # bakes per-node axes, and chained squeeze(0) breaks ORT shape inference.
        depth = output["depth"].squeeze()  # (H, W)
        depth = depth.unsqueeze(0)  # (1, H, W)

        conf = output.get("depth_conf", None)
        if conf is not None:
            conf = conf.squeeze()  # (H, W)
            conf = conf.unsqueeze(0)  # (1, H, W)
            return depth, conf

        return (depth,)


def _squeeze_hw(t: torch.Tensor) -> torch.Tensor:
    """Squeeze all size-1 dims of a depth-like tensor, then re-add batch dim.

    Handles [1,1,H,W], legacy [B,N,1,H,W], and [N,H,W] → [N,H,W].
    """
    out = t.squeeze()
    if out.ndim == 2:
        out = out.unsqueeze(0)
    return out


class AnyViewWrapper(torch.nn.Module):
    """N-view wrapper for multiview/pose/reconstruction/gaussian exports.

    - Input: (N, 3, H, W) batch of pre-processed views.
    - Forwards ``imgs = image.unsqueeze(0)`` → (1, N, 3, H, W) through
      ``model.model(imgs, None, None, [], use_ray_pose, False, "saddle_balanced")``.
    - Best-effort output keys (aliases tried in order; absent heads are simply
      omitted and reported by the driver with a warning):
        depth:        output["depth"]                      → depth [N,H,W]
        conf:         output["depth_conf" | "conf"]        → conf [N,H,W]
        extrinsics:   output["extrinsics" | "pose" | "poses"] → extrinsics [N,3,4]
        intrinsics:   output["intrinsics" | "K" | "focal"]  → intrinsics [N,3,3]
        pointmap:     output["pointmap" | "points" | "point_map"]
        gaussian:     output["gaussian_params" | "gaussians" | "gs_params"]
    """

    def __init__(self, model, task: str, use_ray_pose: bool = False):
        super().__init__()
        self.model = model
        self.task = task
        self.use_ray_pose = use_ray_pose

    @staticmethod
    def _pick(d: dict, *names):
        for n in names:
            if n in d and d[n] is not None:
                return d[n]
        return None

    def forward(self, image: torch.Tensor):
        imgs = image.unsqueeze(0).float()  # (1, N, 3, H, W)
        output = self.model.model(imgs, None, None, [], self.use_ray_pose, False, "saddle_balanced")

        depth = _squeeze_hw(output["depth"])
        outs = [depth]
        names = ["depth"]

        conf = self._pick(output, "depth_conf", "conf")
        if conf is not None:
            outs.append(_squeeze_hw(conf))
            names.append("conf")

        if self.task in ("pose", "reconstruction", "gaussian"):
            extr = self._pick(output, "extrinsics", "pose", "poses")
            if extr is not None:
                if extr.ndim == 4 and extr.shape[0] == 1:
                    extr = extr.squeeze(0)  # [1,N,3,4] -> [N,3,4]
                outs.append(extr)
                names.append("extrinsics")
            intr = self._pick(output, "intrinsics", "K", "focal")
            if intr is not None:
                if intr.ndim == 4 and intr.shape[0] == 1:
                    intr = intr.squeeze(0)  # [1,N,3,3] -> [N,3,3]
                outs.append(intr)
                names.append("intrinsics")

        if self.task in ("reconstruction", "gaussian"):
            pm = self._pick(output, "pointmap", "point_map", "points")
            if pm is not None:
                outs.append(pm)
                names.append("pointmap")

        if self.task == "gaussian":
            gs = self._pick(output, "gaussian_params", "gaussians", "gs_params")
            if gs is not None:
                outs.append(gs)
                names.append("gaussian_params")

        self._last_names = names
        if len(outs) == 1:
            return (outs[0],)
        if len(outs) == 2:
            return outs[0], outs[1]
        if len(outs) == 3:
            return outs[0], outs[1], outs[2]
        if len(outs) == 4:
            return outs[0], outs[1], outs[2], outs[3]
        return tuple(outs)


_CAMERA_IDS = {"d405", "d435", "l515", "kinect", "zed2i-neural", "zed2i-quality", "base"}


def _resolve_model_id(model_name: str) -> str:
    """Resolve --model-name to a Hugging Face repo id.

    Camera-depth ids (d405, d435, l515, kinect, zed2i-neural, zed2i-quality,
    base; case-insensitive) name the separate Camera Depth Model (CDM) family
    in depth-anything/camera-depth-models. Those checkpoints are RGB-D
    (4-channel input), ship as .ckpt/.pth without a DA3 config, and are
    licensed CC BY-NC 4.0. They cannot be exported through this DA3 path, so
    the request fails here with guidance instead of a confusing load error.
    Anything else keeps the legacy depth-anything/<UPPER> mapping.
    """
    slug = model_name.strip().lower()
    if slug in _CAMERA_IDS:
        repo = f"depth-anything/camera-depth-model-{slug}"
        raise ValueError(
            f"'{model_name}' resolves to {repo}, a Camera Depth Model (CDM), not a "
            "Depth Anything 3 checkpoint. CDM models take RGB plus sensor depth "
            "(4-channel input), ship as .ckpt/.pth with no DA3 config, and are "
            "CC BY-NC 4.0; this exporter cannot convert them. Export a DA3 "
            "checkpoint instead (for example --model-name da3-large-1.1), or "
            "port the CDM architecture separately."
        )
    return f"depth-anything/{model_name.upper()}"


def _load_model(model_dir: Path, device: torch.device):
    """Load a model from a Hub id/dir, falling back to manual safetensors load.

    `DepthAnything3.from_pretrained` (HF Hub mixin) expects a
    `pytorch_model.bin`, but upstream repos ship `model.safetensors` +
    `config.json` (with a `model_name` preset key). When the mixin path fails
    and the directory contains those two files, build the preset architecture
    directly and load the safetensors (strict=False; aux heads may be absent).
    """
    from depth_anything_3.api import DepthAnything3

    try:
        model = DepthAnything3.from_pretrained(str(model_dir))
        return model.to(device=device)
    except Exception as first_exc:  # noqa: BLE001
        cfg_path = Path(model_dir) / "config.json"
        st_path = Path(model_dir) / "model.safetensors"
        if not (cfg_path.is_file() and st_path.is_file()):
            raise first_exc
        import json

        preset = json.loads(cfg_path.read_text()).get("model_name")
        if not preset:
            raise first_exc
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise first_exc
        print(f"from_pretrained failed ({first_exc}); falling back to preset "
              f"'{preset}' + manual safetensors load.")
        model = DepthAnything3(preset)
        sd = load_file(str(st_path), device="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"safetensors load: {len(missing)} missing, {len(unexpected)} unexpected keys.")
        return model.to(device=device)


def _warn_missing_heads(task: str, output_names: list) -> None:
    """Warn when the loaded checkpoint lacks heads the requested task wants."""
    if task == "pose" and ("extrinsics" not in output_names or "intrinsics" not in output_names):
        print(
            "WARNING: pose export produced depth-only outputs (no extrinsics/intrinsics). "
            "Use a pose-capable any-view checkpoint (DA3-LARGE/GIANT, not METRIC); "
            "the C++ --task pose falls back to identity poses with a warning.",
            file=sys.stderr,
        )
    if task == "reconstruction" and "pointmap" not in output_names:
        print(
            "WARNING: reconstruction export has no pointmap head; the C++ "
            "--task reconstruction lifts depth maps to output_cloud.ply instead.",
            file=sys.stderr,
        )
    if task == "gaussian" and "gaussian_params" not in output_names:
        print(
            "WARNING: gaussian export has no gaussian_params head (needs the upstream "
            "Gaussian head + gsplat); the C++ --task gaussian writes a depth-lifted "
            "placeholder output_gaussians.ply instead.",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()

    try:
        try:
            from depth_anything_3 import DepthAnything3
        except ImportError:
            # Upstream ships src/depth_anything_3/ without __init__.py
            # (namespace package), so the top-level re-export may not
            # resolve; the class always lives in depth_anything_3.api.
            from depth_anything_3.api import DepthAnything3
    except Exception as exc:  # noqa: BLE001
        print(
            "ERROR: Cannot import depth_anything_3 "
            f"({exc}). Install it with:  pip install -e /path/to/Depth-Anything-3",
            file=sys.stderr,
        )
        return 1

    device = torch.device(args.device)
    print(f"Loading model on {device}…")

    if args.model_dir is not None:
        model = _load_model(args.model_dir, device)
    else:
        try:
            repo = _resolve_model_id(args.model_name)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        model = DepthAnything3.from_pretrained(repo)
        model = model.to(device=device)
    model.eval()

    _patch_cartesian_prod_for_onnx()

    task = args.task
    multiview_tasks = {"multiview", "pose", "reconstruction", "gaussian"}
    is_multiview = task in multiview_tasks
    if is_multiview:
        wrapper = AnyViewWrapper(model, task, use_ray_pose=args.use_ray_pose)
        print(f"Export task: {task} (N-view input, use_ray_pose={args.use_ray_pose})")
    else:
        if args.use_ray_pose:
            print("Note: --use-ray-pose only affects pose-capable (multiview-family) exports; ignored here.")
        wrapper = MonocularWrapper(model)
        print(f"Export task: {task} (single-view input)")
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

    if is_multiview:
        # N=2 representative views: the traced graph carries a dynamic
        # num_views axis on the legacy path (see dynamic_axes below).
        dummy_input = torch.zeros(2, 3, res, res, dtype=torch.float32, device=device)
    else:
        dummy_input = torch.zeros(1, 3, res, res, dtype=torch.float32, device=device)

    print(f"Tracing with input shape: {list(dummy_input.shape)}")

    def _probe_names():
        with torch.no_grad():
            out = wrapper(dummy_input)
        names = getattr(wrapper, "_last_names", None)
        if names is not None:
            return list(names), len(out)
        # MonocularWrapper has no _last_names: depth[, conf].
        return (["depth", "conf"] if len(out) > 1 else ["depth"]), len(out)

    if args.dynamo:
        # Static-shape dynamo export for >2GB models. No dynamic_axes: the
        # traced graph is fixed to the trace resolution anyway (upstream RoPE
        # tables and int()/cache control flow specialize at trace time).
        _patch_rope_for_dynamo()
        output_names, _ = _probe_names()
        print(f"Output names: {output_names}")
        _warn_missing_heads(task, output_names)
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            opset_version=args.opset,
            input_names=["image"],
            output_names=output_names,
            dynamo=True,
        )
    else:
        dynamic_axes: dict | None = None
        if not args.fixed_size:
            if is_multiview:
                dynamic_axes = {
                    "image": {0: "num_views", 2: "height", 3: "width"},
                    "depth": {0: "num_views", 1: "height", 2: "width"},
                }
                output_names, n_out = _probe_names()
                if "conf" in output_names:
                    dynamic_axes["conf"] = {0: "num_views", 1: "height", 2: "width"}
                # Any-view heads carry the leading num_views dim; keep it dynamic
                # so the graph accepts any view count (best-effort by name).
                for extra in ("extrinsics", "intrinsics", "pointmap", "gaussian_params"):
                    if extra in output_names:
                        dynamic_axes[extra] = {0: "num_views"}
            else:
                dynamic_axes = {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "depth": {0: "batch", 1: "height", 2: "width"},
                }
                # Try to detect whether conf output exists
                _, n_out = _probe_names()
                output_names, _ = _probe_names()
                if len(output_names) > 1:
                    dynamic_axes["conf"] = {0: "batch", 1: "height", 2: "width"}
        else:
            output_names, _ = _probe_names()

        print(f"Output names: {output_names}")
        _warn_missing_heads(task, output_names)

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
        if any(getattr(i, "data_location", 0) for i in model_onnx.graph.initializer):
            print("ONNX model with external data loaded successfully; "
                  "skipping in-memory checker (>2GB proto limit).")
        else:
            onnx.checker.check_model(model_onnx)
            print("ONNX model validated successfully.")
    except ImportError:
        print("onnx package not installed — skipping validation.")
    except Exception as exc:  # noqa: BLE001
        print(f"ONNX validation warning: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
