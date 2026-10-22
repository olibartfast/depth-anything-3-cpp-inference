# Requirements — DA3 RealSense Capture

This packet adds native Intel RealSense capture to `inference_app`. It follows the camera-depth
packet: the D435 preset already exists, but the CLI only accepts files, directories, and video
files. This packet closes the gap between "camera on the desk" and "camera-depth checkpoint".

- **Spec:** `specs/2026-10-21-da3-realsense-capture/`
- **Branch:** `spec/da3-realsense-capture`, branched from `master`.
- **Hardware:** Intel RealSense D435 (USB `8086:0b07`), firmware untouched.

## Goal

Capture calibrated color and aligned depth from a RealSense camera inside the C++ binary, then
reuse the existing directory pipeline so every task works without changes to the inference path.

## In scope

- **Optional dependency.** Add `-DUSE_REALSENSE=ON` (default `OFF`). When enabled, CMake finds a
  system `realsense2` package or builds pinned `librealsense` v2.59.1 from source. Default builds,
  Docker, and CI gain no new dependency. (R-1)
- **Native capture command.** Add `--input-realsense N` with `--rs-size WxH` and `--rs-fps K`.
  Capture `N >= 1` frames into a structured directory, then run the existing directory or
  single-image path. No inference-path changes. (R-2)
- **Aligned depth.** Capture the D435 depth stream aligned to the color stream, convert device
  units to millimetres, and store 16-bit PNGs under `depth/`. Color frames go under `color/` so
  `collect_image_paths` never sees depth files. (R-3)
- **Exact intrinsics.** Read color intrinsics from the active stream profile and write
  `intrinsics.txt` in the existing `fx fy cx cy` per-line format. When `--focal` is not passed,
  derive the metric focal from `mean(fx, fy)`; an explicit `--focal` still wins. No intrinsics are
  invented. (R-4)
- **Retention.** Capture into `--capture-dir DIR` when given, otherwise into
  `<output-dir>/realsense_<timestamp>/`. Captures survive the run so depth can be compared later.
  (R-5)
- **Model-free tests.** Put the capture loop behind a `da3::camera::Source` interface and test it
  with a synthetic source: file layout, 16-bit depth encoding, intrinsics formatting, focal
  derivation, and frame-count validation. No camera or model is needed. (R-6)
- **Actionable failures.** A build without RealSense support explains the missing flag; a device
  permission failure mentions the udev rules and the USB node. (R-7)

## Out of scope

- Live preview, continuous inference, and on-screen UI (deferred until batch capture is proven).
- Hardware-depth comparison metrics (abs-rel error, RMSE, delta thresholds). The capture layout
  is the input for that follow-on packet.
- Multi-camera capture, IMU/pose streams, firmware updates, and Windows/macOS capture.
- Automatic model download or checkpoint selection beyond the existing `--camera d435` preset.

## Constraints

- C++20, one inference backend per binary, and `ctest -R UnitTests` runs without hardware.
- librealSense is Apache-2.0; the optional flag keeps the dependency out of the default surface.
- udev rules are required for non-root access. Installation is a documented one-time host step,
  not a build step.
- Capture must not change behavior when `--input-realsense` is absent.

## Definition of done

- R-1 through R-7 are implemented.
- Checks V-1 through V-7 in [validation.md](validation.md) are green with evidence.
- README, `specs/tech-stack.md`, and `specs/roadmap.md` describe the shipped behavior.
