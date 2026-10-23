# TSDF Fusion (`a` / `b` / `c`)

The optional fusion stage turns the frames of a DA3 run into a single
`fused_mesh.ply`. It is implemented by `src/fusion_run.hpp` (mode selection and
dense preprocessing) and `src/fusion_integrator.hpp` (Open3D tensor SLAM or the
dependency-free fallback). The default build stays Open3D-free.

## Build

```sh
# Default: no Open3D, no WITH_FUSION compile definition.
cmake -S . -B build
cmake --build build -j

# Optional TSDF backend.
cmake -S . -B build-fusion -DWITH_FUSION=ON
cmake --build build-fusion -j
```

CMake looks for Open3D with `find_package(Open3D QUIET)`. When a usable
`Open3D::Open3D` (or `Open3D`) target is found, `da3_inference_lib` links it and
exports the `WITH_FUSION` compile definition; only then may the backend include
the Open3D tensor SLAM headers. If Open3D is missing, configuration emits a
`WARNING` and continues with the Open3D-free build. Configuration never fails
for a missing Open3D.

The CLI accepts lowercase mode names only: `--fusion-mode a|b|c` matches
`da3::fusion::parse_fusion_mode()`. Uppercase `A`, `B`, `C` is rejected.

## Modes

| CLI value | Depth source | Trajectory | Scale handling | Needs Open3D |
|---|---|---|---|---|
| `a` | External capture depth from a sibling `depth/` directory; DA3 depth is the per-frame fallback | External `poses.txt` | None; capture depth and poses are already in the capture metric frame | No |
| `b` | DA3 depth (`FrameData::depth`) | External `poses.txt` | Median `ref / pred` alignment against the external capture depth when available | Yes |
| `c` | DA3 depth (`FrameData::depth`) | Model pose head (`FrameData::pose`); identity with a warning when the export has no pose head | None | Yes |

- `a` is the D435 + ICP baseline: external depth plus external poses, no scale
  alignment.
- `b` isolates the depth-scale error: DA3 depth plus the external trajectory,
  with `estimate_scale_median()` applied before integration.
- `c` is model-only: DA3 depth plus the pose tracked by the model.

All three modes are the lowercase `a`/`b`/`c` experiments from the CLI help;
there are no separate uppercase mode names.

## Inputs

The fusion stage consumes the in-memory `tasks::FrameData` already produced by
the same run, after the ordinary task outputs have been written. Frame order is
the input image order and defines the pose row order.

| Input | Used by | Description |
|---|---|---|
| Input images | all | Ordered RGB frames passed to the normal inference path. |
| `poses.txt` | `a`, `b` | One row per frame: 12 floats, row-major **world-to-camera (w2c)** 3x4 `[R \| t]`. Searched in the input directory first, then the output directory. One row must exist per accumulated frame. |
| Sibling `depth/` directory | `a` as depth source, `b` as scale reference | External capture depth next to the image directory (or its parent). 16-bit PNG values are millimetres and are converted to metres; `CV_32F`/`CV_64F` depth is treated as metres. Missing or unreadable frames stay empty: `a` falls back per frame to DA3 depth, and `b` skips alignment for that frame. |
| `FrameData::intrinsics` | all | From `--focal`, or the RealSense device mean focal when auto-filled; rescaled to the actual depth resolution before integration. The runner does not read `intrinsics.txt`. |
| `--dump-raw` output | none | Not required by fusion; the runner works from the postprocessed depth in memory. |

Both text artifacts use the exact `--task pose` formats, so a run remains
reproducible from the ordinary task outputs.

### Pose convention (critical)

Upstream DA3 and `poses.txt` are COLMAP-compatible: poses are **w2c**, i.e.
`p_cam = R * p_world + t`. Open3D `slam::Model` frame poses are
**camera-to-world (c2w)**. `FusionIntegrator::integrate()` therefore converts
every input row once with

```cpp
std::array<float, 12> c2w = da3::fusion::invert_pose_w2c_to_c2w(w2c);
```

before it reaches either backend. Passing raw `poses.txt` rows to Open3D
silently mirrors the trajectory around the world origin and produces a doubled,
smeared mesh. `invert_pose_w2c_to_c2w` computes `R_c2w = R^T` and
`t_c2w = -R^T t`, which is the inverse of the forward projection used elsewhere
in the codebase.

Intrinsics follow the same resolution rule: if the depth-map resolution differs
from the calibration resolution, `scale_intrinsics()` runs first and the scaled
pinhole is used for back-projection and Open3D.

## Preprocess order

The helper order is fixed and shared by all modes:

1. **`scale_intrinsics`** — rescale `fx, fy, cx, cy` from the calibration
   resolution to the depth-map resolution *before* any pixel is interpreted.
2. **`mask_low_confidence`** — zero depth below the confidence threshold
   (default `--conf-thresh 0.5`; pass `0` to keep every sample and rely on
   weighting alone). Modes `b`/`c` supply the model's confidence map, mode `a`
   has no model confidence and leaves depth unchanged.
3. **`compute_valid_mask`** — build the 0/255 mask from the masked depth,
   removing non-positive pixels and Sobel-gradient "flying pixels" above the
   edge threshold (`0.05 m`).
4. **`estimate_scale_median`** — mode `b` only: median of `ref / pred` over
   valid pixels, multiplied into the depth before integration. Returns `1.0`
   when no pixel qualifies.

Confidence also weights integration. The dependency-free fallback stores each
sample with `w = clamp(conf, 0, 1)` and writes the weighted mean per camera ray
and world bin; a deterministic check with two observations of the same pixel
gives `z = 1.9 m` when the 2 m sample has confidence 0.9 and `z = 1.1 m` when
the 1 m sample does. Open3D's pinned tensor SLAM `Frame`/`Model` API exposes no
per-sample confidence or weight tensor, so weighting applies to the fallback
only; the Open3D TSDF backend integrates every valid sample with uniform
weight.

Invalid-mask pixels are zeroed again immediately before integration so edge
artefacts do not become surface points.

## Integration and wiring

`run_offline_fusion()` builds a `FusionIntegrator` with fixed defaults:

```cpp
da3::fusion::FusionIntegrator::Options options;
options.voxel_length = 0.01F;     // metres
options.sdf_trunc = 0.04F;        // metres
options.block_count = 40000;
options.stride = 1;               // fallback: every pixel
options.max_depth_m = 3.0F;       // same 3.0 m upper bound as evaluation
options.max_points = 8000000;     // fallback safety cap

da3::fusion::FusionIntegrator integrator(options);
for (const tasks::FrameData &frame : frames) {
    integrator.integrate(frame.depth, frame.image_bgr, frame.pose, intrinsics);
}
integrator.extract_mesh(fusion_dir / "fused_mesh.ply");
```

Depth is already metric, so the depth-scale contract is fixed:

```cpp
// Depth values are metres, not millimetres.
constexpr float depth_scale = 1.0f;
// Matches the 0.3-3.0 m evaluation window; do not fuse far/noisy surfaces.
constexpr float depth_max = 3.0f;
```

The fallback back-projects valid pixels with the same `depth_max = 3.0` bound.
When wiring Open3D directly, pass the same values rather than the constructor
defaults: `depth_scale = 1.0f`, `depth_max = 3.0f`.

Notes:

- The wrapper owns the w2c -> c2w conversion; do not pre-invert poses at the
  call site.
- `voxel_length` / `sdf_trunc` start at `0.01 m` / `0.04 m` for room-scale
  captures.
- The fallback is intentionally simple: it writes back-projected points as an
  ASCII PLY. It is not a triangulated surface, but the file is a valid
  `*.ply` readable by MeshLab, Open3D, and CloudCompare.

## Outputs

The fusion runner owns exactly one artifact:

```text
<fusion-out>/fused_mesh.ply
```

`<fusion-out>` is `--fusion-out` when set, otherwise `--output-dir`, otherwise
the current directory.

- With Open3D, `fused_mesh.ply` is the extracted triangle mesh.
- In an Open3D-free build, mode `a` writes the dependency-free fallback
  point-cloud PLY under the same filename.

The runner does **not** write, copy, or rewrite `poses.txt` or
`intrinsics.txt`, and it does **not** write or modify `output_cloud.ply`.
Those files stay owned by the ordinary task passes:

| Artifact | Owner |
|---|---|
| `fused_mesh.ply` | fusion runner (this stage) |
| `poses.txt` | `--task pose` caller/pass |
| `intrinsics.txt` | `--task pose` caller/pass |
| `output_cloud.ply` | `--task reconstruction` caller/pass |

Keeping the ordinary outputs untouched makes fusion additive: a run can emit a
point cloud and a fused mesh side by side for comparison.

## Open3D-free behaviour

- `--fusion-mode a` always works. In a `WITH_FUSION=OFF` build it uses the
  fallback backend and reports that `fused_mesh.ply` is the fallback
  point-cloud PLY.
- `--fusion-mode b` and `--fusion-mode c` require the Open3D TSDF backend. In
  an Open3D-free build (or a `-DWITH_FUSION=ON` configure that did not find a
  usable Open3D target), the CLI reports:

  ```text
  Fusion mode 'b' requires a build with -DWITH_FUSION=ON and a usable Open3D install
  ```

  and exits non-zero *before* loading the model. It never silently degrades to
  the fallback for these modes.

## Evaluation

Evaluate every mode with the same frames, masks, and bounds:

- **Depth accuracy:** `da3::fusion::abs_rel(pred_m, ref_m, valid, 0.3, 3.0)` —
  mean absolute relative error over valid pixels with reference depth in the
  **0.3-3.0 m** window. Pixels outside the window or invalid in the mask are
  skipped. Report per mode (`a`/`b`/`c`) and per frame.
- **Trajectory:** `da3::fusion::ate_translation_rmse(est_c2w, ref_c2w)` — RMSE
  of translation differences in metres. Feed c2w arrays to both arguments; a
  w2c/c2w mix is the most common source of a bogus multi-metre ATE.
- **Geometry:** vertex/triangle counts, bounding box, and a fixed-view render
  or point sample to spot double walls and holes. A mode `c` mesh should have
  fewer isolated components and a thinner wall band than mode `b`.
- **Determinism:** rerun mode `c` twice; identical inputs must produce the same
  vertex count (Open3D integration is deterministic for a fixed frame order).

Acceptance: mode `a` runs with `WITH_FUSION=OFF` and produces the fallback PLY;
modes `b`/`c` configure cleanly with Open3D, and mode `c` yields a finite ATE
and a mesh whose extent is within the 0.3-3.0 m evaluation range of the
reference.

## Non-goals

- No Open3D types leak into public headers; `fusion_utils.hpp` stays
  dependency-free and Open3D code is confined to the translation unit behind
  `WITH_FUSION`.
- No automatic loop closure, global optimization, or learned fusion.
- No change to the DA3 export path; this stage only consumes its outputs.
