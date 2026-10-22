# Validation — DA3 RealSense Capture

Validation is defined before implementation. Hardware checks run only when the D435 is connected
and accessible; otherwise the row is recorded as `BLOCKED` with the exact reason.

| ID | Requirement | Check | Pass condition |
|----|-------------|-------|----------------|
| V-1 | R-1 optional dependency | Configure and build without `-DUSE_REALSENSE`; inspect the link line and `--help` | Default build succeeds with no librealsense on the link line; `--input-realsense` reports that the binary was built without support |
| V-2 | R-2 capture command | `inference_app model.onnx --input-realsense 3 ...` | Exactly three color frames are captured and the existing directory path runs |
| V-3 | R-3 aligned depth | Inspect `capture_dir/color/000.png` and `capture_dir/depth/000.png` | Depth PNG is 16-bit, same dimensions as color, and non-zero over the scene; color directory contains no depth images |
| V-4 | R-4 exact intrinsics | Compare `intrinsics.txt` with the device profile; run with and without `--focal` | Captured `fx fy cx cy` are positive and plausible for the chosen size; focal defaults to `mean(fx, fy)`; explicit `--focal` wins |
| V-5 | R-5 retention | Run with and without `--capture-dir` | Explicit directory keeps frames and depth; default directory is `realsense_<timestamp>` under the output directory |
| V-6 | R-6 model-free tests | `ctest -R UnitTests` on the default build | Capture tests pass with a synthetic source; no camera or model is required |
| V-7 | R-7 actionable failures | Run the non-RealSense build and a permission-denied hardware build | Messages name the missing build flag and the udev-rules remedy respectively |

## Evidence

| Date | Check | Result | Notes |
|------|-------|--------|-------|
| 2026-10-21 | V-1 | PASS | Default Release build links no librealsense; `--input-realsense` reports "requires a binary built with -DUSE_REALSENSE=ON" |
| 2026-10-21 | V-6 | PASS | `ctest -R UnitTests` green with `test_camera_capture.cpp` (synthetic source, no camera or model) |
| 2026-10-22 | V-1 (hardware build) | PASS | `-DUSE_REALSENSE=ON -DWERROR=ON` builds librealsense v2.59.1 from source (SDK at C++17), links `build-rs/inference_app`, and compiles the project targets with no warnings; `--help` lists the capture flags |
| 2026-10-22 | V-6 (hardware build) | PASS | `ctest --test-dir build-rs -R UnitTests` green; the capture tests still need no device |
| 2026-10-22 | V-7 (no device) | PASS | With the camera unplugged, the app prints "No device connected" and the udev-rules remedy |
| 2026-10-22 | V-2..V-5 | BLOCKED | D435 enumerates (`RealSense D435`, serial 215322071944, firmware 5.15.0.2) and librealsense reads color intrinsics at 640x480 (`fx=606.861 fy=606.818 cx=318.18 cy=251.563`), but the link is USB 2.1: UVC frames are truncated (65536 of 614655 bytes, "Incomplete frame received"), and OpenCV reads one frame from `/dev/video6` before timing out. Move the camera to a USB 3 port and rerun |
| 2026-10-22 | V-2 (USB3) | PASS | D435 on USB 3.20 (`lsusb` `bcdUSB 3.20`, sysfs speed 5000 Mbps, serial 218423027387); `inference_app tests/fixtures/tiny_depth.onnx --input-realsense 3 --capture-dir /tmp/opencode/p4_evidence/v2` exited 0, reported "Captured 3 color frame(s) and 3 aligned depth frame(s)", and the existing directory pipeline processed all three `color/*.png` |
| 2026-10-22 | V-2..V-5 (warm-up fix) | PASS | The first USB3 run captured near-black color (mean 0.01–7, max 28) because the color auto-exposure had not converged. `src/realsense_source.cpp` now discards 30 warm-up frames after the pipeline starts; later runs captured stable color (mean 80.9–83.7, max ~205) while still writing exactly the requested three frames |
| 2026-10-22 | V-3 (USB3) | PASS | `depth/{000,001,002}.png` are 1280x720 16-bit PNGs, same dimensions as color and aligned to it; 83.7–84.0% of depth pixels are non-zero (0–~3.5 m in millimetres). `color/` contains only `000.png`, `001.png`, `002.png`; no depth images are written there |
| 2026-10-22 | V-4 (USB3) | PASS | Color profile intrinsics at 1280x720: `fx=910.292 fy=910.227 cx=637.271 cy=377.344` (all positive, centered near 640x360). Without `--focal` the app printed "Focal from device intrinsics: 910.259 px (mean of fx and fy)"; with `--focal 123.5` that auto-fill line was absent and the run still exited 0, so the explicit value wins (`main.cpp` gates auto-fill on `!focal_set`) |
| 2026-10-22 | V-5 (USB3) | PASS | Run with `--output-dir /tmp/opencode/p4_evidence/v5out` and no `--capture-dir` created `realsense_1792692027030650002/` containing `color/` (3 PNGs), `depth/` (3 PNGs), `intrinsics.txt`, and `capture.txt`; an explicit `--capture-dir /tmp/opencode/p4_evidence/v2` retained the same three color and depth frames |
| 2026-10-22 | V-2, V-3, V-5 (USB3 re-validation) | PASS | D435 serial 218423027387 on USB 3.20 (`lsusb -d 8086:0b07 -v`: `bcdUSB 3.20`); rebuilt `build-rs` from the current tree and ran `inference_app tests/fixtures/tiny_depth.onnx --input-realsense 3 --capture-dir /tmp/opencode/p4_v2` → exit 0, "Captured 3 color frame(s) and 3 aligned depth frame(s)", `color/` holds exactly `000..002.png`. Depth PNGs are 1280x720 `uint16` matching color, 83.4–83.6% non-zero (0.25–10.5 m), and no depth image is written under `color/`. The explicit directory kept all three frames; a run with `--output-dir /tmp/opencode/p4_v5out` and no `--capture-dir` created `realsense_1792693978001040137/` with `color/`, `depth/`, `intrinsics.txt`, and `capture.txt` |
| 2026-10-22 | V-4 (USB3 re-validation) | PASS | `intrinsics.txt` rows are all `910.292 910.227 637.271 377.344` at 1280x720 (positive, centred near 640x360). Without `--focal` the app printed "Focal from device intrinsics: 910.259 px (mean of fx and fy)"; with `--focal 123.5` that line is absent and, under `--task metric`, `output_metric_000.npy` range changed from `[-5.898, 7.185]` to `[-0.800, 0.980]` (≈7.33x, near the 910.259/123.5 = 7.37x focal ratio; verified 2026-10-23, scene-dependent), so the explicit value wins |
| 2026-10-22 | V-6 (both builds) | PASS | Default `build/` (`-DUSE_REALSENSE=OFF -DWERROR=ON`) rebuilt with no warnings, `ldd build/inference_app` shows no librealsense, and `ctest -R UnitTests` is green; hardware `build-rs` (`-DUSE_REALSENSE=ON -DWERROR=ON`) also passes `ctest -R UnitTests` with the synthetic capture tests and no device or model dependency |
