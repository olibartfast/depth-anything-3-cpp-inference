# Video Depth Inference with TensorRT

This guide batches monocular depth over a video clip using the TensorRT (TRT) backend. It was proven on 2026-10-12 on an RTX 3060 (`sm_86`) with TensorRT 11.2.1.2 and a fixed-size `da3-small` model.

The examples use `/tmp/da3vid2` for frames and outputs. Replace that path with your own working directory.

## Prerequisites

- A TensorRT build:
  ```bash
  cmake -S . -B build-trt -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON
  cmake --build build-trt
  ```
  Set `TENSORRT_ROOTDIR` when using a pre-staged TensorRT prefix.
- The TensorRT library directory, CUDA 13 `cudart`/`cublas`, and cuDNN 9 on `LD_LIBRARY_PATH`.
- A fixed-size model. TensorRT engines are fixed-shape and reject dynamic ONNX graphs:
  ```bash
  python export_onnx.py --model-name da3-small --output da3-small-fixed.onnx --fixed-size
  ```
- `ffmpeg` with PNG/JPG and libx264 support.

## 1. Prepare the source clip

Any clip works. The proven run used a clean Pexels clip with no overlays: <https://www.pexels.com/video/people-walking-on-the-street-8996518> (HD 720×1280 portrait, 30 fps), fetched from <https://videos.pexels.com/video-files/8996518/8996518-hd_720_1280_30fps.mp4>.

The fixed 504×504 engine needs a square input. Aspect-preserving preprocessing maps a square to exactly the engine shape, so center-crop to a square and scale it. This example takes the first five seconds (150 frames):

```bash
mkdir -p /tmp/da3vid2/frames /tmp/da3vid2/depth
ffmpeg -y -i pexels-clean.mp4 -t 5 \
  -vf "crop=720:720:0:280,scale=504:504" /tmp/da3vid2/frames/f%04d.png
```

Choose the crop window that covers the subject:

| Clip orientation | Crop expression | Example |
|---|---|---|
| Portrait, `w×H` | `crop=<w>:<w>:0:<y>` | `crop=720:720:0:280` |
| Landscape, `W×h` | `crop=<h>:<h>:<x>:0` | `crop=720:720:280:0` |

## 2. Run batch inference

The engine is built once, then reused from disk:

```bash
export LD_LIBRARY_PATH=<trt-lib>:<cudart13>:<cudnn>:$LD_LIBRARY_PATH
cd /tmp/da3vid2
i=0
for f in frames/f*.png; do
  i=$((i + 1)); n=$(printf "%04d" "$i")
  <repo>/build-trt/inference_app <repo>/da3-small-fixed.onnx "$f" --process-res 504 \
    > last.log 2>&1 || { echo "FAIL $f"; tail -3 last.log; break; }
  mv -f output_depth.jpg "depth/d${n}.jpg"
done
```

Notes:

- The first run builds `da3-small-fixed.engine` next to the ONNX file and can take minutes. Later runs, or runs that pass the `.engine` file as the model argument, deserialize it in about 1–2 seconds.
- The output is JPEG (`output_depth.jpg`). Keep the `.jpg` suffix when renaming; a `.png` pattern breaks the assembly step.
- The measured cost is about 1.4 seconds per frame, dominated by per-process engine load. The model itself runs in milliseconds; an in-app video mode would amortize the load.

## 3. Reassemble the depth video

Set the output frame rate to match the source. The proven run used 30 fps:

```bash
ffmpeg -y -r 30 -i depth/d%04d.jpg -pix_fmt yuv420p /tmp/da3vid2/depth_trt.mp4
```

## 4. Build a side-by-side video

Build the square RGB clip with the same crop and scale, then stack it horizontally with the depth clip:

```bash
ffmpeg -y -i <clip>.mp4 -t 5 -vf "crop=720:720:0:280,scale=504:504" \
  -pix_fmt yuv420p /tmp/da3vid2/rgb504.mp4
ffmpeg -y -i /tmp/da3vid2/rgb504.mp4 -i /tmp/da3vid2/depth_trt.mp4 \
  -filter_complex hstack -pix_fmt yuv420p /tmp/da3vid2/side_by_side.mp4
```

The command matches the portrait example. Substitute the crop expression for your clip; the landscape variant is in the crop table above.

## Proven result

The run processed 150 of 150 frames in 211 seconds. The outputs were `/tmp/da3vid2/depth_trt.mp4` (504×504, 30 fps, 5 seconds) and `/tmp/da3vid2/side_by_side.mp4` (1008×504).

In the depth video, foreground legs resolve dark purple, the mid-crowd sits in mid-tones, and the sky is bright. The boardwalk shows a clean near-to-far gradient, and the silhouettes align with the RGB positions.

Measured on the RTX 3060 (`inference_app` prints `Timing:` for each run):

- Initialization: about 740 ms for the one-time engine deserialize.
- Inference: about 190–206 ms per frame, including host-to-device (H2D) transfer, GPU execution, and device-to-host (D2H) transfer in FP32.

The YouTube render (`youtube.mp4`, 2016×1008, `+faststart`) burns in the title, RGB and depth labels, the 195 ms/frame figure, and the model and footage credits. See `docs/video-side-by-side.jpg`.

For a calmer upload variant (`youtube_4fps.mp4`, 37.5 seconds at 30 fps), each inferred frame is held for 250 ms using `setpts=7.5*PTS`. This duplicates frames without motion-blur interpolation; an earlier `minterpolate` 0.5× attempt was discarded because of the blur. The depth label notes the 250 ms/frame cadence.
