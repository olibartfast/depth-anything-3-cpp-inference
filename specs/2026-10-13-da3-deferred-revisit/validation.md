# Validation — DA3 Deferred Revisit

The evidence was collected on 2026-10-13 on an RTX 3060 Laptop. The requirements are in [requirements.md](requirements.md).

Terms used below: 3DGS is 3D Gaussian Splatting; gsplat is the CUDA Gaussian-splatting library; MJPG is Motion JPEG; PLY is the Polygon File Format.

## Video input

**V-1 — passed.** `--input-video t.avi (5 frames at 10 fps, MJPG) --fps 2 --task multiview` reported “Extracted 1 frame(s)”, wrote the depth JPG, and removed the temporary directory; the absence was verified after the run. The implemented path uses `cv::VideoCapture` rather than ffmpeg because `videoio` was already available and no new dependency was needed. The `VideoExtract.SamplesByFps` unit test passed: 5 frames at 10 fps sampled at 5 fps produced 3 frames, a rate of 0 produced all 5, and a missing input threw. Exclusivity between `--input-video` and a positional path or `--input-dir`, plus empty-input acceptance, were also tested.

`--process-res 280` on the 504-trace dynamic model fails with the expected F-1 broadcast error, which is documented. The video path itself is shape-agnostic.

## 3DGS head

**V-2 — BLOCKED with a dated reason.** gsplat was not installed. It is a CUDA-extension source build, which is disproportionate on a 6 GB laptop, and no Gaussian-head checkpoint was available: DA3-SMALL exposes no `gaussian_params` head. The exporter warns about this, and the C++ path writes the documented placeholder instead. Revisit when Gaussian-head weights and a CUDA build machine are available. The `--task gaussian` export slot (`gaussian_params` output name) and the C++ reader path are already in place.

## Keep-or-drop decisions

**V-3 — completed.** The verdicts below are outcomes, not deferrals.

| Item | Verdict | Rationale |
|---|---|---|
| Video input | **KEPT (shipped)** | `--input-video` and `--fps` use `cv::VideoCapture` with no new dependencies. `videoio` was already available on the system and was added to the CMake components. |
| Binary PLY | **DROPPED (stride suffices)** | `--ply-stride` solves the size problem: 373 KB at stride 4. Binary PLY would add reader complexity without a validated user need. |
| glTF export | **DROPPED** | PLY, `poses.txt`, and depth `.npy` cover interchange, and there is no renderer in the C++ scope. |
| 3DGS head (gsplat) | **CONDITIONAL (see V-2)** | The placeholder path shipped. The full head needs weights and a gsplat build. |
| SLAM-Long | **DROPPED** | Long-video SLAM is an external system. Chunked streaming covers the C++-side need. |
| Gradio parity | **DROPPED** | Gradio is a Python application layer. The mission is the C++ runtime. |
| DA3-BENCH | **DROPPED** | Benchmarking tooling for research is explicitly not the C++ runtime. |
| Giant and Nested weights | **NOT ATTEMPTED** | CC BY-NC 4.0 licensing and more than 6 GB of VRAM at 504 exceed this machine. Small proves the full any-view path. |
