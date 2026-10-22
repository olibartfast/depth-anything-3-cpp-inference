# Plan — DA3 RealSense Capture

The work is split into three thin stages. Acceptance checks are in [validation.md](validation.md).

## Stage A1 — Capture core (no hardware)

Add `src/camera_capture.hpp` and `src/camera_capture.cpp` with the hardware-free pieces:

- `da3::camera::Intrinsics` and `focal_px()` (mean of `fx`, `fy`).
- `da3::camera::Source`, a blocking `next(Frame&)` interface.
- `capture_to_directory()`, which writes `color/%03d.png`, `depth/%03d.png` (16-bit millimetres),
  `intrinsics.txt`, and `capture.txt`.

Add `tests/unit/test_camera_capture.cpp` with a synthetic source. Writable paths:
`src/camera_capture.{hpp,cpp}`, `tests/unit/test_camera_capture.cpp`, `CMakeLists.txt`.

## Stage A2 — RealSense backend and CMake

Add `src/realsense_source.{hpp,cpp}` behind `USE_REALSENSE`:

- `rs2::pipeline` with BGR8 color and Z16 depth at the requested size and rate.
- `rs2::align(RS2_STREAM_COLOR)` for depth→color alignment.
- Intrinsics from the active color `video_stream_profile`.
- Depth conversion `raw * depth_units * 1000` to millimetres.

CMake: `option(USE_REALSENSE OFF)`, system-package lookup first, pinned source build second, and a
public `USE_REALSENSE` compile definition. Writable paths: `src/realsense_source.{hpp,cpp}`,
`CMakeLists.txt`.

## Stage A3 — CLI, docs, and evidence

Wire `--input-realsense N`, `--rs-size WxH`, `--rs-fps K`, and `--capture-dir DIR` into
`main.cpp`. Capture before inference, point the existing pipeline at `color/`, auto-fill the focal
from the device when `--focal` is absent, and print the device name. Update README,
`specs/tech-stack.md`, and `specs/roadmap.md`. Writable paths: `src/main.cpp`, `README.md`,
`specs/tech-stack.md`, `specs/roadmap.md`.

## Deviations

- **SDK standard.** librealsense v2.59.1 does not compile under GCC 13 with C++20
  (an injected-class-name constructor in `rsutils`). CMake builds only the SDK
  subproject at C++17 and restores C++20 for the application. Clang 18 accepts
  the code, but the pin keeps both compilers working.

## Validation references

V-1 covers R-1, V-2 covers R-2, V-3 covers R-3, V-4 covers R-4, V-5 covers R-5, V-6 covers R-6,
and V-7 covers R-7.

## Integration

After A1–A3:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests

# Optional hardware build
cmake -S . -B build-rs -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_REALSENSE=ON
cmake --build build-rs --parallel
./build-rs/inference_app model.onnx --input-realsense 3 --camera d435 --task metric \
  --capture-dir /tmp/rs_capture
```

The default build must stay dependency-free. The hardware build must either capture three frames
or stop with the udev hint.
