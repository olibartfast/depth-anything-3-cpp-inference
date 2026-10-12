# Video Depth Inference (TensorRT) — Replication Guide

Batch monocular depth on a video clip using the TensorRT backend.
Proven 2026-10-12 on RTX 3060 (sm_86), TRT 11.2.1.2, `da3-small` fixed-size model.

## Prereqs

- TRT build: `cmake -S . -B build-trt -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON && cmake --build build-trt`
  (`TENSORRT_ROOTDIR` set when using a pre-staged prefix)
- Runtime env: TRT lib dir + CUDA 13 cudart/cublas + cuDNN 9 on `LD_LIBRARY_PATH`
- Fixed-size model (TRT engine is fixed-shape — dynamic ONNX is rejected):
  `python export_onnx.py --model-name da3-small --output da3-small-fixed.onnx --fixed-size`
- `ffmpeg` with PNG/JPG + libx264

## 1. Source clip

Any clip works. The proven run used a clean Pexels clip (no overlays):
`https://www.pexels.com/video/people-walking-on-the-street-8996518`
(HD 720×1280 portrait, 30 fps), fetched via the direct file link
`https://videos.pexels.com/video-files/8996518/8996518-hd_720_1280_30fps.mp4`.
The fixed 504×504 engine needs **square** input
(aspect-preserving preprocess maps square → exactly the engine shape), so
center-crop to square and scale (first 5 s, 150 frames):

```bash
mkdir -p /tmp/da3vid2/frames /tmp/da3vid2/depth
ffmpeg -y -i pexels-clean.mp4 -t 5 \
  -vf "crop=720:720:0:280,scale=504:504" /tmp/da3vid2/frames/f%04d.png
```

(Adjust the `crop` to your clip: `crop=<w>:<w>:0:<y>` for portrait `w×H`,
`crop=<h>:<h>:<x>:0` for landscape. Pick the window covering the subjects.)

```bash
mkdir -p /tmp/da3vid/frames /tmp/da3vid/depth
ffmpeg -y -i <clip>.mp4 -t 5 \
  -vf "crop=720:720:280:0,scale=504:504" /tmp/da3vid/frames/f%04d.png
```

(Adjust the `crop` to your clip: `crop=<h>:<h>:<x>:0` for landscape `W×h`.)

## 2. Batch inference (engine reused from disk after first build)

```bash
export LD_LIBRARY_PATH=<trt-lib>:<cudart13>:<cudnn>:$LD_LIBRARY_PATH
cd /tmp/da3vid
i=0
for f in frames/f*.png; do
  i=$((i + 1)); n=$(printf "%04d" "$i")
  ./build-trt/inference_app da3-small-fixed.onnx "$f" --process-res 504 \
    > last.log 2>&1 || { echo "FAIL $f"; tail -3 last.log; break; }
  mv -f output_depth.jpg "depth/d${n}.jpg"
done
```

Notes:
- First run builds `da3-small-fixed.engine` next to the ONNX (minutes);
  later runs (or `.engine` as model arg) deserialize it in ~1–2 s.
- `output_depth.jpg` is JPEG — keep the `.jpg` suffix (PNG pattern breaks assembly).
- Cost is ~1.4 s/frame, dominated by per-process engine load; the model
  itself runs in milliseconds (in-app video mode would amortize this).

## 3. Reassemble depth video

```bash
ffmpeg -y -r 25 -i depth/d%04d.jpg -pix_fmt yuv420p /tmp/da3vid/depth_trt.mp4
```

## 4. Side-by-side (RGB | depth)

Build the square RGB clip with the same crop/scale, then stack:

```bash
ffmpeg -y -i <clip>.mp4 -t 5 -vf "crop=720:720:280:0,scale=504:504" \
  -pix_fmt yuv420p /tmp/da3vid/rgb504.mp4
ffmpeg -y -i /tmp/da3vid/rgb504.mp4 -i /tmp/da3vid/depth_trt.mp4 \
  -filter_complex hstack -pix_fmt yuv420p /tmp/da3vid/side_by_side.mp4
```

## Proven result

150/150 frames in 211 s, `/tmp/da3vid2/depth_trt.mp4` (504×504, 30 fps, 5 s),
`/tmp/da3vid2/side_by_side.mp4` (1008×504). Foreground legs resolve dark
purple, mid crowd mid-tones, sky bright; boardwalk shows a clean near→far
gradient; silhouettes align with RGB positions.
