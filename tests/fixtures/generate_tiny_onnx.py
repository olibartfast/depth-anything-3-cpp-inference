#!/usr/bin/env python3
"""Generate the tiny ONNX fixture used by the ONNX Runtime backend tests.

The graph is intentionally trivial: it returns the per-pixel channel mean,
shape [1, H, W] from input [1, 3, H, W]. It exercises ONNX Runtime session
creation, dynamic spatial dimensions, tensor I/O, and output extraction without
shipping model weights.

Requires `onnx`:
    python3 tests/fixtures/generate_tiny_onnx.py
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

OUTPUT = Path(__file__).with_name("tiny_depth.onnx")


def build_model() -> onnx.ModelProto:
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "height", "width"])
    depth = helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, "height", "width"])

    reduce_mean = helper.make_node(
        "ReduceMean",
        inputs=["image"],
        outputs=["depth"],
        axes=[1],
        keepdims=0,
    )

    graph = helper.make_graph(
        [reduce_mean],
        "tiny_depth_channel_mean",
        [image],
        [depth],
    )
    model = helper.make_model(
        graph,
        producer_name="depth-anything-3-cpp-inference tests",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    # ONNX Runtime 1.21 supports IR v10; pin an older, widely supported IR.
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> int:
    model = build_model()
    onnx.save(model, OUTPUT)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
