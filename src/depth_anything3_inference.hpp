#pragma once
#include "backends/inference_backend.hpp"

#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Bring backend namespace into scope for convenience
using da3::backend::create_backend;
using da3::backend::InferenceBackend;

/// Configuration for the Depth Anything 3 inference engine.
struct Config {
    /// Target resolution for the longest side of the input image (pixels).
    /// Images will be resized so the longest side equals this value, then
    /// rounded to the nearest multiple of patch_size.
    int process_res{504};

    /// DINOv2 patch size — output spatial dimensions must be divisible by this.
    int patch_size{14};

    /// Whether to save and display the confidence map in addition to the depth map.
    bool output_conf{false};

    /// ImageNet normalization statistics.
    std::array<float, 3> means{0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds{0.229f, 0.224f, 0.225f};
};

/// Result of a single-image depth estimation inference.
struct DepthResult {
    /// Depth map with shape [H, W], in model-native units (relative by default).
    cv::Mat depth;

    /// Confidence map with shape [H, W], values in [0, 1].
    /// Empty when Config::output_conf is false or the model has no conf output.
    cv::Mat conf;

    /// Original image width (before any resize).
    int orig_w{0};

    /// Original image height (before any resize).
    int orig_h{0};
};

/// C++ inference wrapper for Depth Anything 3 monocular depth estimation.
///
/// The typical workflow:
///   1. Construct with a model path (ONNX or TensorRT engine) and an optional Config.
///   2. Call preprocess_image() to get the CHW float tensor.
///   3. Call run_inference() with the tensor.
///   4. Call postprocess_outputs() to retrieve the DepthResult.
///   5. Call colorize_depth() / save_output_image() as needed.
class DepthAnything3Inference {
  public:
    /// Primary constructor — loads and initializes the model from disk.
    explicit DepthAnything3Inference(const std::filesystem::path &model_path, const Config &config = Config{});

    /// Test-friendly constructor — injects a custom backend (skips I/O).
    DepthAnything3Inference(std::unique_ptr<InferenceBackend> backend, const Config &config = Config{});

    ~DepthAnything3Inference() = default;

    // Non-copyable, non-movable (owns unique backend)
    DepthAnything3Inference(const DepthAnything3Inference &) = delete;
    DepthAnything3Inference &operator=(const DepthAnything3Inference &) = delete;
    DepthAnything3Inference(DepthAnything3Inference &&) = delete;
    DepthAnything3Inference &operator=(DepthAnything3Inference &&) = delete;

    /// Preprocess the input image from a file path.
    /// @param image_path  Path to a BGR/RGB image file readable by OpenCV.
    /// @param orig_h      Filled with the original image height.
    /// @param orig_w      Filled with the original image width.
    /// @param out_h       Filled with the model-input height (after resize & patch alignment).
    /// @param out_w       Filled with the model-input width.
    /// @returns Flattened CHW float tensor, size = 3 * out_h * out_w.
    std::vector<float> preprocess_image(const std::filesystem::path &image_path, int &orig_h, int &orig_w, int &out_h,
                                        int &out_w);

    /// Preprocess the input image from an in-memory cv::Mat (avoids disk I/O).
    std::vector<float> preprocess_image(const cv::Mat &bgr_image, int &orig_h, int &orig_w, int &out_h, int &out_w);

    /// Run inference on a preprocessed CHW float tensor.
    void run_inference(std::span<const float> input_data, int out_h, int out_w);

    /// Extract depth (and optionally confidence) from the last inference run.
    /// @param out_h / out_w  Model output spatial dimensions (same as passed to run_inference).
    /// @param orig_h / orig_w  Original image dimensions (for optional upscaling).
    /// @param upscale_to_orig  If true, resize the depth map back to the original image resolution.
    [[nodiscard]] DepthResult postprocess_outputs(int out_h, int out_w, int orig_h, int orig_w,
                                                  bool upscale_to_orig = true);

    /// Produce a visualizable BGR depth colormap from a raw depth cv::Mat.
    [[nodiscard]] static cv::Mat colorize_depth(const cv::Mat &depth_map);

    /// Produce a visualizable BGR confidence colormap from a raw conf cv::Mat.
    [[nodiscard]] static cv::Mat colorize_conf(const cv::Mat &conf_map);

    /// Save an image to the specified path.
    /// @returns The saved path on success, std::nullopt on failure.
    [[nodiscard]] static std::optional<std::filesystem::path> save_output_image(const cv::Mat &image,
                                                                                const std::filesystem::path &path);

    // -------------------------------------------------------------------------
    // Accessors (mostly for testing)
    // -------------------------------------------------------------------------
    [[nodiscard]] int get_process_res() const noexcept { return config_.process_res; }
    [[nodiscard]] int get_patch_size() const noexcept { return config_.patch_size; }
    [[nodiscard]] const Config &get_config() const noexcept { return config_; }

  private:
    std::unique_ptr<InferenceBackend> backend_;
    Config config_;
    std::vector<int64_t> input_shape_;

    // Cached output tensors from the last run_inference() call
    std::vector<std::vector<float>> output_data_cache_;
    std::vector<std::vector<int64_t>> output_shapes_cache_;
};
