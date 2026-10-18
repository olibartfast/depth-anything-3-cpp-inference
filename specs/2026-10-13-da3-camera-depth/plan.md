# Plan — DA3 Camera-Depth and Export Shape-Lock

The work is split into two thin stages. The acceptance checks are in [validation.md](validation.md).

## Stage A1 — Camera preset and checkpoint mapping

Add `parse_camera`, `camera_to_string`, and `camera_checkpoint` to the header. Add `Options.camera` and the `--camera` parse path to `main.cpp`. The `base` preset implies the metric-mode default focal behavior; the other presets still apply focal scaling. An explicit `--focal` wins.

Writable paths: `src/depth_anything3_inference.hpp` and `src/main.cpp`.

## Stage A2 — Export, documentation, and tests

Extend `export_onnx.py` with camera model IDs in the `--model-name` help and with the shape-lock warning text. Add the README camera table and the R-4 unit tests.

Writable paths: `export_onnx.py`, `README.md`, and `tests/unit/test_depth_anything3_inference.cpp`.

## Validation references

Acceptance is defined in [validation.md](validation.md): V-1 covers R-1, V-2 covers R-2, V-3 covers R-3 and F-1, V-4 covers R-4, and V-5 covers R-5.

## Integration

After A1 and A2:

```bash
cmake --build build --parallel
ctest --test-dir build -R UnitTests
./build/inference_app --help
python3 export_onnx.py --help
```

The `--help` output must show the camera presets, and the export help must describe the trace-shape lock and `--fixed-size` guidance.
