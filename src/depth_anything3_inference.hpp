#pragma once
#include "backends/inference_backend.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Bring backend namespace into scope for convenience
using da3::backend::create_backend;
using da3::backend::InferenceBackend;

/// Native Depth Anything 3 tasks selectable with `--task`.
///
/// These mirror the upstream `da3` CLI modes. Only `mono`/`metric` have
/// complete single-graph support in this binary today; the remaining tasks are
/// accepted by the CLI and degrade to depth-only output (see requirements D-4).
enum class TaskType { mono, metric, multiview, pose, reconstruction, gaussian };

/// Parse a `--task` argument into a TaskType.
/// @param name  Raw task name; an empty string maps to TaskType::mono.
/// @throws std::runtime_error if the name is not a recognized task.
[[nodiscard]] inline TaskType parse_task(const std::string &name) {
    if (name.empty() || name == "mono") {
        return TaskType::mono;
    }
    if (name == "metric") {
        return TaskType::metric;
    }
    if (name == "multiview") {
        return TaskType::multiview;
    }
    if (name == "pose") {
        return TaskType::pose;
    }
    if (name == "reconstruction") {
        return TaskType::reconstruction;
    }
    if (name == "gaussian") {
        return TaskType::gaussian;
    }
    throw std::runtime_error("Unknown task: '" + name +
                             "' (expected one of: mono, metric, multiview, pose, reconstruction, gaussian)");
}

/// Canonical, lowercase CLI name for a TaskType (inverse of parse_task).
[[nodiscard]] inline std::string task_to_string(TaskType task) {
    switch (task) {
    case TaskType::mono:
        return "mono";
    case TaskType::metric:
        return "metric";
    case TaskType::multiview:
        return "multiview";
    case TaskType::pose:
        return "pose";
    case TaskType::reconstruction:
        return "reconstruction";
    case TaskType::gaussian:
        return "gaussian";
    }
    return "mono";
}

/// Recommended upstream checkpoints for a task (surfaced in `--help`).
[[nodiscard]] inline std::string recommended_checkpoint(TaskType task) {
    switch (task) {
    case TaskType::mono:
        return "DA3-SMALL/BASE/LARGE/GIANT";
    case TaskType::metric:
        return "DA3METRIC-LARGE";
    case TaskType::multiview:
        return "DA3-LARGE/GIANT";
    case TaskType::pose:
        return "DA3-LARGE/GIANT (not METRIC)";
    case TaskType::reconstruction:
        return "DA3-GIANT; NESTED-GIANT-LARGE for metric";
    case TaskType::gaussian:
        return "DA3-LARGE/GIANT + Gaussian head";
    }
    return "";
}

/// Camera preset for per-camera metric depth checkpoints from the Hugging Face
/// collection `depth-anything/camera-depth-models`.
///
/// These checkpoints are trained per camera model and are metric-oriented:
/// they output depth that needs the camera focal length (px) to reach meters
/// (see the metric formula in README/`processing_utils`). No fx/fy values are
/// baked in — the exact intrinsics remain a user `--focal` input (R-2).
enum class CameraPreset { none, d405, d435, l515, kinect, zed2i_neural, zed2i_quality, base };

/// Parse a `--camera` argument into a CameraPreset.
/// @param name  Raw camera id; an empty string maps to CameraPreset::none.
/// @throws std::runtime_error if the name is not a recognized camera preset.
[[nodiscard]] inline CameraPreset parse_camera(const std::string &name) {
    if (name.empty()) {
        return CameraPreset::none;
    }
    if (name == "d405") {
        return CameraPreset::d405;
    }
    if (name == "d435") {
        return CameraPreset::d435;
    }
    if (name == "l515") {
        return CameraPreset::l515;
    }
    if (name == "kinect") {
        return CameraPreset::kinect;
    }
    if (name == "zed2i-neural") {
        return CameraPreset::zed2i_neural;
    }
    if (name == "zed2i-quality") {
        return CameraPreset::zed2i_quality;
    }
    if (name == "base") {
        return CameraPreset::base;
    }
    throw std::runtime_error("Unknown camera preset: '" + name +
                             "' (expected one of: d405, d435, l515, kinect, zed2i-neural, zed2i-quality, base)");
}

/// Canonical, lowercase CLI name for a CameraPreset (inverse of parse_camera).
/// The no-preset sentinel maps to the empty string.
[[nodiscard]] inline std::string camera_to_string(CameraPreset preset) {
    switch (preset) {
    case CameraPreset::none:
        return "";
    case CameraPreset::d405:
        return "d405";
    case CameraPreset::d435:
        return "d435";
    case CameraPreset::l515:
        return "l515";
    case CameraPreset::kinect:
        return "kinect";
    case CameraPreset::zed2i_neural:
        return "zed2i-neural";
    case CameraPreset::zed2i_quality:
        return "zed2i-quality";
    case CameraPreset::base:
        return "base";
    }
    return "";
}

/// Recommended checkpoint id within `depth-anything/camera-depth-models` for a
/// preset. The no-preset sentinel maps to the empty string. Kept separate from
/// camera_to_string so the CLI name and the HF checkpoint id can diverge later
/// without touching callers.
[[nodiscard]] inline std::string camera_checkpoint(CameraPreset preset) {
    switch (preset) {
    case CameraPreset::none:
        return "";
    case CameraPreset::d405:
        return "d405";
    case CameraPreset::d435:
        return "d435";
    case CameraPreset::l515:
        return "l515";
    case CameraPreset::kinect:
        return "kinect";
    case CameraPreset::zed2i_neural:
        return "zed2i-neural";
    case CameraPreset::zed2i_quality:
        return "zed2i-quality";
    case CameraPreset::base:
        return "base";
    }
    return "";
}

/// Fully-qualified Hugging Face repo for a camera preset, in
/// `depth-anything/camera-depth-model-<checkpoint>` form (verified against the
/// HF collection page 2026-10-13). The no-preset sentinel maps to "".
[[nodiscard]] inline std::string camera_hf_id(CameraPreset preset) {
    const std::string checkpoint = camera_checkpoint(preset);
    if (checkpoint.empty()) {
        return "";
    }
    return "depth-anything/camera-depth-model-" + checkpoint;
}

/// Whether a camera preset implies `--already-metric` semantics for the given
/// focal-override state.
///
/// This is a CLI-defined rule (spec R-1), not an upstream claim: the generic
/// `base` preset is treated as already-metric by default, so the
/// `focal*net/300` scaling is skipped. An explicit `--focal` always wins.
[[nodiscard]] inline bool camera_implies_already_metric(CameraPreset preset, bool focal_set) noexcept {
    return preset == CameraPreset::base && !focal_set;
}

/// Collect image paths (*.jpg, *.jpeg, *.png — case-insensitive) from a
/// directory, sorted lexicographically for deterministic multi-view ordering.
/// Returns an empty vector when @p dir is not a readable directory.
[[nodiscard]] inline std::vector<std::filesystem::path> collect_image_paths(const std::filesystem::path &dir) {
    std::vector<std::filesystem::path> paths;
    if (!std::filesystem::is_directory(dir)) {
        return paths;
    }
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

/// File-based outputs for the pose / reconstruction / gaussian tasks.
///
/// Lives in the header (not main.cpp) so unit tests exercise the real writers.
/// Pose heads are matched by trailing shape because the backend interface
/// exposes output indices + shapes, not names (see find_matrix_output in
/// main.cpp): [3,4] trailing dims = extrinsics, [3,3] = intrinsics.
namespace tasks {

/// One processed frame accumulated for multi-frame task artifacts.
struct FrameData {
    cv::Mat depth;                     // postprocessed depth (metric-scaled when applicable)
    cv::Mat image_bgr;                 // original BGR image (colour source for PLY outputs)
    std::array<float, 12> pose{};      // 3x4 world-to-camera, row-major: r11 r12 r13 tx r21 …
    std::array<float, 4> intrinsics{}; // fx fy cx cy
    bool pose_from_model{false};       // true when the ONNX exposed an extrinsics head
};

/// Default pinhole focal length (px) when --focal is not passed: 1.2 * width.
[[nodiscard]] inline float default_focal(int width) noexcept {
    return 1.2f * static_cast<float>(width);
}

/// Back-project a depth pixel with the pinhole model and transform by the
/// frame pose: X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy then p_world = R * p + t.
inline void backproject_point(const FrameData &frame, int u, int v, float z, float &out_x, float &out_y,
                              float &out_z) {
    const float fx = frame.intrinsics[0] > 0.0f ? frame.intrinsics[0] : 1.0f;
    const float fy = frame.intrinsics[1] > 0.0f ? frame.intrinsics[1] : 1.0f;
    const float cam_x = (static_cast<float>(u) - frame.intrinsics[2]) * z / fx;
    const float cam_y = (static_cast<float>(v) - frame.intrinsics[3]) * z / fy;
    const float cam_z = z;

    out_x = frame.pose[0] * cam_x + frame.pose[1] * cam_y + frame.pose[2] * cam_z + frame.pose[3];
    out_y = frame.pose[4] * cam_x + frame.pose[5] * cam_y + frame.pose[6] * cam_z + frame.pose[7];
    out_z = frame.pose[8] * cam_x + frame.pose[9] * cam_y + frame.pose[10] * cam_z + frame.pose[11];
}

/// Resize the frame's colour image to the (postprocessed) depth size.
[[nodiscard]] inline cv::Mat frame_color_at_depth_size(const FrameData &frame) {
    if (frame.image_bgr.empty() || frame.depth.empty()) {
        return {};
    }
    cv::Mat color;
    cv::resize(frame.image_bgr, color, cv::Size(frame.depth.cols, frame.depth.rows), 0, 0, cv::INTER_LINEAR);
    return color;
}

/// Write N poses, one whitespace-separated 12-float row per line:
///   r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
[[nodiscard]] inline bool write_poses_txt(const std::filesystem::path &path,
                                          const std::vector<std::array<float, 12>> &poses) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << std::setprecision(9);
    for (const auto &pose : poses) {
        for (std::size_t k = 0; k < 12; ++k) {
            if (k != 0) {
                out << ' ';
            }
            out << pose[k];
        }
        out << '\n';
    }
    return static_cast<bool>(out);
}

/// Write N intrinsics rows, one `fx fy cx cy` per line.
[[nodiscard]] inline bool write_intrinsics_txt(const std::filesystem::path &path,
                                               const std::vector<std::array<float, 4>> &intrinsics) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << std::setprecision(6);
    for (const auto &k : intrinsics) {
        out << k[0] << ' ' << k[1] << ' ' << k[2] << ' ' << k[3] << '\n';
    }
    return static_cast<bool>(out);
}

/// Back-project every frame's depth map into an ASCII PLY point cloud.
/// COUNT = H*W*N over all non-empty frames.
/// @param stride  Pixel subsampling (1 = every pixel; 2 = every 2nd row/col).
[[nodiscard]] inline bool save_point_cloud_ply(const std::filesystem::path &path,
                                               const std::vector<FrameData> &frames, int stride = 1) {
    const int step = stride > 0 ? stride : 1;
    std::size_t total = 0;
    for (const auto &frame : frames) {
        if (!frame.depth.empty()) {
            total += static_cast<std::size_t>((frame.depth.rows + step - 1) / step) *
                     static_cast<std::size_t>((frame.depth.cols + step - 1) / step);
        }
    }

    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << total << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n";
    out << std::setprecision(6);

    for (const auto &frame : frames) {
        if (frame.depth.empty()) {
            continue;
        }
        const cv::Mat color = frame_color_at_depth_size(frame);
        for (int v = 0; v < frame.depth.rows; v += step) {
            const float *depth_row = frame.depth.ptr<float>(v);
            for (int u = 0; u < frame.depth.cols; u += step) {
                float x = 0.0f;
                float y = 0.0f;
                float z = 0.0f;
                backproject_point(frame, u, v, depth_row[u], x, y, z);

                int red = 0;
                int green = 0;
                int blue = 0;
                if (!color.empty()) {
                    const cv::Vec3b &px = color.at<cv::Vec3b>(v, u);
                    blue = px[0];
                    green = px[1];
                    red = px[2];
                }
                out << x << ' ' << y << ' ' << z << ' ' << red << ' ' << green << ' ' << blue << '\n';
            }
        }
    }
    return static_cast<bool>(out);
}

/// Write a per-pixel Gaussian PLY placeholder:
///   x y z nx ny nz red green blue opacity scale_0..2 rot_0..rot_3
/// opacity=1, scales from the local depth gradient (fallback 0.01*Z),
/// normal +Z, rotation = identity quaternion.
[[nodiscard]] inline bool save_gaussians_ply(const std::filesystem::path &path,
                                             const std::vector<FrameData> &frames, int stride = 1) {
    const int gstep = stride > 0 ? stride : 1;
    std::size_t total = 0;
    for (const auto &frame : frames) {
        if (!frame.depth.empty()) {
            total += static_cast<std::size_t>((frame.depth.rows + gstep - 1) / gstep) *
                     static_cast<std::size_t>((frame.depth.cols + gstep - 1) / gstep);
        }
    }

    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << total << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property float nx\n"
        << "property float ny\n"
        << "property float nz\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "property float opacity\n"
        << "property float scale_0\n"
        << "property float scale_1\n"
        << "property float scale_2\n"
        << "property float rot_0\n"
        << "property float rot_1\n"
        << "property float rot_2\n"
        << "property float rot_3\n"
        << "end_header\n";
    out << std::setprecision(6);

    for (const auto &frame : frames) {
        if (frame.depth.empty()) {
            continue;
        }
        const int h = frame.depth.rows;
        const int w = frame.depth.cols;
        const cv::Mat color = frame_color_at_depth_size(frame);

        for (int v = 0; v < h; v += gstep) {
            const int v0 = std::max(0, v - 1);
            const int v1 = std::min(h - 1, v + 1);
            for (int u = 0; u < w; u += gstep) {
                const float z = frame.depth.at<float>(v, u);
                float x = 0.0f;
                float y = 0.0f;
                float zc = 0.0f;
                backproject_point(frame, u, v, z, x, y, zc);

                const int u0 = std::max(0, u - 1);
                const int u1 = std::min(w - 1, u + 1);
                const float dzdx = (frame.depth.at<float>(v, u1) - frame.depth.at<float>(v, u0)) * 0.5f;
                const float dzdy = (frame.depth.at<float>(v1, u) - frame.depth.at<float>(v0, u)) * 0.5f;
                float scale = std::sqrt(dzdx * dzdx + dzdy * dzdy);
                if (!(scale > 1e-6f)) {
                    scale = 0.01f * std::fabs(z);
                }
                if (!(scale > 1e-6f)) {
                    scale = 1e-4f;
                }

                int red = 0;
                int green = 0;
                int blue = 0;
                if (!color.empty()) {
                    const cv::Vec3b &px = color.at<cv::Vec3b>(v, u);
                    blue = px[0];
                    green = px[1];
                    red = px[2];
                }

                out << x << ' ' << y << ' ' << zc << ' ' << 0.0f << ' ' << 0.0f << ' ' << 1.0f << ' ' << red << ' '
                    << green << ' ' << blue << ' ' << 1.0f << ' ' << scale << ' ' << scale << ' ' << scale << ' '
                    << 1.0f << ' ' << 0.0f << ' ' << 0.0f << ' ' << 0.0f << '\n';
            }
        }
    }
    return static_cast<bool>(out);
}

} // namespace tasks

/// Configuration for the Depth Anything 3 inference engine.
struct Config {    /// Target resolution for the longest side of the input image (pixels).
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

    /// Run inference on N preprocessed CHW frames stacked as an [N,3,H,W] batch.
    /// @param frames  N CHW tensors, each of size 3*out_h*out_w.
    /// @throws std::invalid_argument when frames is empty or sizes differ.
    void run_inference_batch(const std::vector<std::vector<float>> &frames, int out_h, int out_w);

    /// Extract depth (and optionally confidence) from the last inference run.
    /// @param out_h / out_w  Model output spatial dimensions (same as passed to run_inference).
    /// @param orig_h / orig_w  Original image dimensions (for optional upscaling).
    /// @param upscale_to_orig  If true, resize the depth map back to the original image resolution.
    [[nodiscard]] DepthResult postprocess_outputs(int out_h, int out_w, int orig_h, int orig_w,
                                                  bool upscale_to_orig = true);

    /// Per-view slice of a batched run: same as postprocess_outputs but reads
    /// view @p view from [N,H,W] outputs cached by run_inference_batch().
    /// Falls back to the single output when the model returned [1,H,W].
    [[nodiscard]] DepthResult postprocess_view(size_t view, int out_h, int out_w, int orig_h, int orig_w,
                                               bool upscale_to_orig = true);

    /// Per-view pose extraction from the cached outputs.
    /// Matches [N,3,4] trailing dims for extrinsics and [N,3,3] for intrinsics
    /// (the backend exposes indices + shapes, not names); @p pose is always
    /// filled (identity fallback), @p intrinsics only overridden on a match.
    /// @returns true when an extrinsics head was found.
    [[nodiscard]] bool get_pose_view(size_t view, std::array<float, 12> &pose,
                                     std::array<float, 4> &intrinsics) const;

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

    /// Model input shape detected at initialize() time: [1,3,0,0] placeholder
    /// for dynamic models, concrete [N,3,H,W] for static ones (e.g. dynamo
    /// any-view exports). Drives the C++ N-view batch path + F-1 shape check.
    [[nodiscard]] const std::vector<int64_t> &get_model_input_shape() const noexcept { return input_shape_; }

    /// Raw output tensors (data + shapes) cached by the last run_inference() call.
    /// Additive, read-only accessor used by the pose/point-map tasks, which need
    /// model outputs beyond depth and confidence (e.g. a [3,4] extrinsics head or
    /// a [3,3] intrinsics head). Both vectors are empty before the first
    /// run_inference() call; a backend that exposes no such outputs simply yields
    /// no matching shape, and callers fall back to identity pose.
    [[nodiscard]] const std::vector<std::vector<float>> &get_raw_outputs() const noexcept {
        return output_data_cache_;
    }
    [[nodiscard]] const std::vector<std::vector<int64_t>> &get_raw_output_shapes() const noexcept {
        return output_shapes_cache_;
    }

  private:
    std::unique_ptr<InferenceBackend> backend_;
    Config config_;
    std::vector<int64_t> input_shape_;

    // Cached output tensors from the last run_inference() call
    std::vector<std::vector<float>> output_data_cache_;
    std::vector<std::vector<int64_t>> output_shapes_cache_;
};
