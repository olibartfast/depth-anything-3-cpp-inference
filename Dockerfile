# ONNX Runtime Dockerfile for Depth Anything 3 C++ Inference
# Multi-stage build to keep final image small

# --- Build stage ---
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    clang-18 \
    libopencv-dev \
    wget \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/clang-18 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
    -DUSE_ONNX_RUNTIME=ON \
    -DUSE_TENSORRT=OFF \
    -DWERROR=ON

RUN cmake --build build --parallel

# --- Runtime stage ---
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core406t64 \
    libopencv-imgproc406t64 \
    libopencv-imgcodecs406t64 \
    libopencv-highgui406t64 \
    libopencv-videoio406t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/build/inference_app /usr/local/bin/inference_app
# Keep one copy of the ORT library and recreate the soname links that the
# release tarball ships as duplicate files.
COPY --from=builder /workspace/build/_deps/onnxruntime-linux-x64-*/lib/libonnxruntime.so.1.21.0 /usr/local/lib/
RUN ln -sf libonnxruntime.so.1.21.0 /usr/local/lib/libonnxruntime.so.1 \
    && ln -sf libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so \
    && ldconfig

ENTRYPOINT ["inference_app"]
CMD ["--help"]

# Usage:
# Build: docker build -t da3-onnx .
# Run:   docker run -v $(pwd)/data:/data da3-onnx /data/model.onnx /data/image.jpg
