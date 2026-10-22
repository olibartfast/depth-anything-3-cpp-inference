# Plan — Fusion Validation and Tracking

Five thin stages. Each ends with a runnable check; hardware work comes last so code defects
are found before a capture session. Acceptance checks are in [validation.md](validation.md).

## Stage F1 — Focal provenance and camera-input audit

Split the focal flag in `src/main.cpp`:

- Keep `focal_set` as "user passed `--focal`"; add `focal_from_device` for the RealSense
  auto-fill. `Options::skip_metric_scaling()` consults only the user flag, so
  `camera_implies_already_metric()` behaves as documented for `--camera base`.
- Verify the `d435` checkpoint input signature against upstream `depth_anything_3` source
  (read the model wrapper, do not infer from the collection page). Record the result in
  `specs/2026-10-13-da3-camera-depth/validation.md` as a new row.

Writable: `src/main.cpp`, `tests/unit/test_depth_anything3_inference.cpp`,
`specs/2026-10-13-da3-camera-depth/validation.md`.

Check: unit test asserts `base` plus auto-filled focal skips scaling, and explicit `--focal`
still scales.

## Stage F2 — Pose provider and ICP tracker

Add `src/fusion_pose.hpp` (header-only, OpenCV only):

- `PoseProvider` interface returning w2c rows.
- `ExternalPoseProvider` wrapping `read_poses_txt`.
- `ModelPoseProvider` wrapping the accumulated `FrameData::pose` values, with an explicit
  `has_model_poses` flag so identity fallback cannot masquerade as tracking.
- `IcpPoseProvider`: point-to-plane ICP against the previous depth frame or the raycast model,
  three pyramid levels, projective association, Huber weighting, residual rejection.
  Returns `std::nullopt` when it cannot converge; the runner must then fail loudly instead of
  continuing with a stale pose.

Add unit tests with a synthetic depth ramp and a known rigid transform: recovered rotation and
translation within tolerance, and a deliberate outlier scene that must be rejected.

Writable: `src/fusion_pose.hpp`, `tests/unit/test_depth_anything3_inference.cpp`.

Check: synthetic transform recovered within 1 mm and 0.1 degrees.

## Stage F3 — Confidence weighting

- Extend `FusionIntegrator::Options` with a weighting mode and accept the per-pixel `conf` map
  alongside depth.
- Fallback path: accumulate `w` per point and divide at extract time.
- Open3D path: pass the weight tensor where the pinned release exposes it; otherwise document
  the limitation in `docs/fusion.md` rather than silently dropping confidence.
- Keep the binary mask behaviour available (`--conf-thresh`) because it bounds flying pixels.

Writable: `src/fusion_integrator.hpp`, `src/fusion_run.hpp`, `src/main.cpp`.

Check: a synthetic two-frame scene with one high-confidence and one low-confidence surface
weights the fused result toward the confident surface.

## Stage F4 — Open3D pin and CI leg

- Pin the current stable Open3D release; record version and commit in `specs/tech-stack.md`.
- Document the exact `slam::Model` method the release exposes (`ProcessFrame` versus
  `Integrate`) and confirm the existing constrained dispatch compiles against it.
- Add a CI job that installs the pinned Open3D and builds `-DWITH_FUSION=ON -DWERROR=ON`.
  Keep the default job Open3D-free.
- Fix the wording everywhere: fallback output is a point cloud, not a TSDF mesh.

Writable: `.github/workflows/ci.yml`, `CMakeLists.txt`, `specs/tech-stack.md`, `docs/fusion.md`,
`src/fusion_integrator.hpp` only if the pinned API forces a change.

Check: `-DWITH_FUSION=ON` builds and links in CI; `fused_mesh.ply` from that build is a
triangle mesh, not points.

## Stage F5 — Hand-held A/B/C session

Capture and evaluate:

```bash
# 1. Capture a moving sequence (translate and rotate; do not pan from a tripod)
./build-rs/inference_app <model.onnx> --input-realsense 60 --task metric \
  --capture-dir captures/d435_handheld --output-dir captures/d435_handheld/da3

# 2. Mode a: sensor depth + ICP poses, then b and c with the same trajectory
./build-rs/inference_app <model.onnx> --fusion-mode a --capture-dir ... 
```

- Require a trajectory with measured baseline: print total path length and reject runs below
  0.2 m. A static run must fail rather than report numbers.
- Evaluate each mode with `da3::fusion::chamfer_l1`, `eval_ate_translation_rmse`, and
  `eval_abs_rel`; hold the reference fixed (sensor depth where valid) across modes.
- Record: frame count, path length, per-mode Chamfer/ATE/abs_rel, and wall-clock per frame in
  `validation.md`. Report failures as failures.

Writable: `specs/2026-10-23-da3-fusion-validation/validation.md`, `docs/fusion.md`, `README.md`.

Check: three modes have numbers on one committed command set; if Open3D is unavailable, mode
results come from the fallback and that is stated explicitly.

## Deviations

Record every deviation here with a date. If Open3D cannot be pinned in CI within budget, mark
R-4 `BLOCKED` with the exact reason and keep the fallback path; do not claim mesh validation.

## Integration

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DWERROR=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests

cmake -S . -B build-rs -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_REALSENSE=ON -DWERROR=ON
cmake --build build-rs --parallel
```

Default build stays dependency-free; the hardware build captures and the fusion modes run.
