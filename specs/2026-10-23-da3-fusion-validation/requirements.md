# Requirements — Fusion Validation and Tracking

This packet turns the shipped fusion front end into a validated pipeline. The TSDF integration
code, evaluation metrics, and RealSense capture already exist, but no experiment has ever run
A/B/C on a moving camera, no pose tracker exists, and the Open3D backend has never compiled
against a real Open3D install. It closes those gaps in that order.

- **Spec:** `specs/2026-10-23-da3-fusion-validation/`
- **Branch:** `spec/da3-fusion-validation`, branched from `master`.
- **Predecessor:** [any-view fusion](../2026-10-13-da3-anyview-fusion/requirements.md) (front end),
  [RealSense capture](../2026-10-21-da3-realsense-capture/requirements.md) (depth source).

## Goal

Produce measured A/B/C evidence on a hand-held D435 sequence: fused TSDF mesh quality, pose
error, and depth error, each reproducible from committed commands. Fix the defects that block
that measurement and nothing else.

## In scope

- **Focal provenance.** Separate `focal_from_user` from `focal_from_device` so
  `camera_implies_already_metric()` sees only a user-specified focal. Today the RealSense
  auto-fill sets `focal_set = true`, so `--camera base --input-realsense` scales already-metric
  depth by about `mean(fx,fy)/500`. Add a CLI-level test that `base` skips scaling after
  auto-fill and that an explicit `--focal` still wins. (R-1)
- **Pose source abstraction.** Add `da3::fusion::PoseProvider` with three implementations:
  `ExternalPoseProvider` (reads `poses.txt`), `ModelPoseProvider` (consumes the DA3 pose head),
  and `IcpPoseProvider` (frame-to-model on an existing TSDF or depth frame). Each returns
  `std::array<float, 12>` in **w2c**; inversion stays where it is today, in the integrator.
  (R-2)
- **ICP tracker.** Implement frame-to-model ICP for mode `a` and as the mode `c` fallback:
  point-to-plane, projective data association, coarse-to-fine pyramid, and a rejection threshold
  on the residual. It consumes the D435 aligned depth, not DA3 depth, so the baseline is
  independent of the model. No new dependency: OpenCV plus the existing fusion helpers.
  (R-3)
- **Open3D backend verification.** Pin one Open3D release (current stable at implementation
  time), document the exact `slam::Model` entry point it exposes, and add a CI leg that
  configures `-DWITH_FUSION=ON` against a prebuilt Open3D. The dependency-free fallback must
  keep emitting a point cloud under the `.ply` name; documentation must not call that a mesh.
  (R-4)
- **Confidence-weighted integration.** Weight each depth sample by `conf` instead of a binary
  mask: `w = clamp(conf, 0, 1)` in the fallback path and the Open3D weight tensor where the
  pinned release supports it. Document the mapping when it does not. (R-5)
- **Hand-held A/B/C run.** Record a short D435 sequence (about 30–60 frames, translation and
  rotation, no pure pan), then run: `a` sensor depth plus ICP poses, `b` DA3 depth plus the same
  ICP poses plus median rescale, `c` DA3 depth plus model poses. Report Chamfer distance,
  ATE, and `abs_rel` over 0.3–3.0 m for each mode. A static sequence is explicitly not
  acceptable evidence: at zero baseline fusion collapses to denoising. (R-6)
- **Camera-depth input check.** Verify against upstream source whether the `d435`-style
  checkpoints accept RGB only or RGB plus sensor depth, and record the finding. If they accept
  sensor depth, wire that input; if not, document it and close the question. No guessing.
  (R-7)

## Out of scope

- Learned tracking (DROID-SLAM, DPVO), neural or Gaussian SLAM, feed-forward reconstruction
  (DUSt3R, MASt3R, VGGT), and semantics. These stay outside the mission.
- Fisheye or non-pinhole camera models.
- Real-time operation. The 195 ms/frame offline budget stands.
- 3DGS head, long-sequence SLAM, video reconstruction, and Gradio parity (settled in the
  [deferred revisit](../2026-10-13-da3-deferred-revisit/validation.md)).

## Constraints

- C++20; default build stays Open3D-free and RealSense-free.
- `-DWERROR=ON` clean on gcc and clang; clang-tidy and clang-format gates green.
- Model-free unit tests keep passing; no weights or captures committed.
- Pose convention never changes: `poses.txt` is w2c, inversion only in `invert_pose_w2c_to_c2w`.
- Every reported number must come from a committed command with its output captured.

## Definition of done

- R-1 through R-7 are implemented or `BLOCKED` with a dated reason.
- V-1 through V-7 in [validation.md](validation.md) are green with evidence, including one
  hand-held sequence and per-mode metrics.
- `docs/fusion.md` states which modes are validated and which remain unverified.
