# Validation — DA3 Deferred Revisit (Phase D)

## Evidence (2026-10-13, RTX 3060 laptop)
- V-1 video: `--input-video t.avi (5f@10fps, MJPG) --fps 2 --task multiview` → "Extracted 1 frame(s)", depth jpg written, temp dir removed (verified absent after run). Unit `VideoExtract.SamplesByFps` green (5f@10fps @5fps→3 frames, @0→5, missing→throw). Exclusivity (`--input-video` + positional/--input-dir) and empty-input acceptance tested. `--process-res 280` on the 504-trace dyn model fails with the F-1 broadcast error (expected, documented) — video path itself is shape-agnostic.
- V-2 gsplat probe: NOT INSTALLED (decision, not oversight). Rationale: gsplat is a CUDA-extension build (heavy on a 6GB laptop), and no Gaussian-head checkpoint is on hand (DA3-SMALL has no `gaussian_params` head — the exporter already warns + the C++ writes the documented placeholder). The day a `+Gaussian head` checkpoint is available, the `--task gaussian` export path + C++ reader slot (`gaussian_params` output name) are already in place. Revisit trigger: weights in hand.
- V-3 keep/drop decisions:

| Item | Verdict | Rationale |
|------|---------|-----------|
| Video input | KEPT (shipped) | `--input-video/--fps` via `cv::VideoCapture`, no new deps (videoio already on system; added to CMake components) |
| Binary PLY | DROPPED (stride suffices) | `--ply-stride` solves the size problem (373KB @stride-4); binary PLY adds reader complexity for no validated user need |
| glb export | DROPPED | PLY + poses.txt + depth npy cover interchange; no renderer in C++ scope |
| 3DGS head (gsplat) | CONDITIONAL (see V-2) | Placeholder path shipped; full head needs weights + gsplat build |
| SLAM-Long | DROPPED | External long-video system; chunked streaming covers the C++-side need |
| Gradio parity | DROPPED | Python app layer; mission is the C++ runtime |
| DA3-BENCH | DROPPED | Research benchmarking tooling, explicitly not the C++ runtime (mission) |
| Giant/Nested weights | NOT ATTEMPTED | CC BY-NC 4.0 licensing + >6GB VRAM at 504 exceed this machine; Small proves the full any-view path |
