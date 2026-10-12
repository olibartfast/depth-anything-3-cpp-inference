#pragma once

#ifdef USE_TENSORRT

#include "inference_backend.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <memory>

// Only TensorRT 11 is supported (same pin as rf-detr-cpp-inference:
// versions.env TENSORRT_VERSION=11.2.1.2, CUDA 13.3).
#if NV_TENSORRT_MAJOR < 11
#error "TensorRT 11 or newer is required"
#endif

namespace da3::backend {

// Custom deleter for TensorRT objects (TRT 11 uses plain delete)
struct TensorRTDeleter {
    template <typename T> void operator()(T *obj) const {
        if (obj) {
            delete obj;
        }
    }
};

/**
 * @brief TensorRT implementation of InferenceBackend
 *
 * This backend uses NVIDIA TensorRT for optimized GPU inference.
 * Provides low-latency and high-throughput inference on NVIDIA GPUs.
 *
 * TensorRT 11 is strongly typed: engine precision follows the ONNX graph,
 * so FP16 comes from converting the ONNX beforehand (no kFP16 flag).
 * The ONNX model must have static shapes — export with
 * `export_onnx.py --fixed-size` for the TensorRT path (dynamic spatial
 * dims are rejected at build time with a clear error).
 */
class TensorRTBackend : public InferenceBackend {
  public:
    TensorRTBackend();
    ~TensorRTBackend() override;
    TensorRTBackend(const TensorRTBackend &) = delete;
    TensorRTBackend &operator=(const TensorRTBackend &) = delete;
    TensorRTBackend(TensorRTBackend &&) = delete;
    TensorRTBackend &operator=(TensorRTBackend &&) = delete;

    std::vector<int64_t> initialize(const std::filesystem::path &model_path,
                                    const std::vector<int64_t> &input_shape) override;

    void run_inference(std::span<const float> input_data, const std::vector<int64_t> &input_shape) override;

    [[nodiscard]] size_t get_output_count() const override;

    void get_output_data(size_t output_index, float *data, size_t size) override;

    [[nodiscard]] std::vector<int64_t> get_output_shape(size_t output_index) const override;

    [[nodiscard]] std::string get_backend_name() const override { return "TensorRT"; }

  private:
    // Binds every IO tensor address and enqueues on stream_. Does not synchronise.
    void enqueue();

    // TensorRT logger
    class Logger : public nvinfer1::ILogger {
      public:
        void log(Severity severity, const char *msg) noexcept override;
    };

    // Build TensorRT engine from ONNX model
    bool build_engine_from_onnx(const std::filesystem::path &model_path, const std::vector<int64_t> &input_shape);

    // Serialize and save engine to file
    void serialize_engine(const std::filesystem::path &engine_path);

    // Deserialize engine from file
    bool deserialize_engine(const std::filesystem::path &engine_path);

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TensorRTDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRTDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRTDeleter> context_;

    // CUDA buffers
    std::vector<void *> device_buffers_;
    std::vector<std::vector<float>> host_output_buffers_;

    // Dedicated non-blocking stream. Everything — the host path's copies and
    // the enqueue — runs here rather than on the legacy default stream, which
    // serialises against the whole process.
    cudaStream_t stream_{nullptr};

    // Tensor metadata
    std::vector<std::vector<int64_t>> output_shapes_;
    size_t input_element_count_ = 0;
    int input_binding_index_ = -1;
    std::vector<int> output_binding_indices_;
};

} // namespace da3::backend

#endif // USE_TENSORRT
