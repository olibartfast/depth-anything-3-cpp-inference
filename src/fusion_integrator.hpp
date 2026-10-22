#pragma once

// Header-only TSDF/point-cloud fusion facade for the DA3 reconstruction stage.
//
// Two implementations share one public interface:
//
//   * WITH_FUSION builds that can see the Open3D tensor SLAM API use
//     `open3d::t::pipelines::slam::Model` for real volumetric integration and
//     triangle-mesh extraction.
//   * Every other build (the default) uses a dependency-free fallback that
//     back-projects valid depth pixels to world space (OpenCV + STL only).
//     `extract_mesh()` then writes the accumulated points as a valid ASCII PLY
//     file; it is a point cloud rather than a triangulated surface, but it is a
//     well-formed `*.ply` readable by MeshLab/Open3D/CloudCompare.
//
// Pose convention
// ---------------
// DA3/COLMAP poses are stored world-to-camera (w2c): p_cam = R * p_world + t.
// Open3D `slam::Model` frame poses (and this class's trajectory accessor) are
// camera-to-world (c2w). `integrate()` therefore calls
// `invert_pose_w2c_to_c2w()` on every input pose *before* any Open3D call, and
// the fallback path transforms points with that same c2w matrix so both
// backends agree. `trajectory_c2w()` returns the converted poses.
//
// Thread safety: a single instance is not internally synchronized. Use one
// instance per thread or serialize calls externally.

#include "fusion_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <system_error>
#include <utility>
#include <vector>

// Probe for the Open3D tensor SLAM surface. The full set of headers (Model,
// Frame, and tensor IO) is required so the selected backend can both integrate
// and export a mesh; if any piece is missing we stay on the fallback path.
#if defined(WITH_FUSION) && defined(__has_include)
#if __has_include(<open3d/t/pipelines/slam/Model.h>) && __has_include(<open3d/t/pipelines/slam/Frame.h>) &&                 \
    __has_include(<open3d/core/Tensor.h>) && __has_include(<open3d/t/io/TriangleMeshIO.h>) &&                              \
    __has_include(<open3d/t/io/PointCloudIO.h>)
#define DA3_FUSION_USE_OPEN3D 1
#include <open3d/core/Tensor.h>
#include <open3d/t/io/PointCloudIO.h>
#include <open3d/t/io/TriangleMeshIO.h>
#include <open3d/t/pipelines/slam/Frame.h>
#include <open3d/t/pipelines/slam/Model.h>
#endif
#endif

namespace da3::fusion {

namespace detail {

/// Dependent false for static_asserts that must only fire on instantiation.
template <typename> inline constexpr bool dependent_false = false;

} // namespace detail

/// Incremental depth/colour fusion with a c2w trajectory history.
///
/// Typical use:
///   FusionIntegrator integrator;
///   for (frame : frames) integrator.integrate(frame.depth, frame.bgr, frame.pose_w2c, k);
///   integrator.extract_mesh("fused.ply");
///   auto c2w = integrator.trajectory_c2w();
class FusionIntegrator {
  public:
    /// Tuning knobs shared by both backends.
    struct Options {
        /// Open3D TSDF voxel edge length (metres). Ignored by the fallback.
        float voxel_length{0.01F};
        /// Open3D TSDF truncation distance (metres). Ignored by the fallback.
        float sdf_trunc{0.04F};
        /// Open3D preallocated spatial blocks. Ignored by the fallback.
        std::int64_t block_count{40000};
        /// Pixel subsampling for the fallback path (1 = every pixel).
        int stride{1};
        /// Upper depth bound in metres for the fallback back-projection.
        /// Defaults to 3.0 m to match the 0.3-3.0 m evaluation window used by
        /// eval_abs_rel() / abs_rel(..., 0.3, 3.0), so fused geometry and the
        /// reported metrics see the same depth range. Values <= 0 disable it.
        float max_depth_m{3.0F};
        /// Safety cap on fallback points; values <= 0 mean unlimited.
        std::size_t max_points{8000000};
    };

    FusionIntegrator() = default;

    explicit FusionIntegrator(const Options &options) : options_(options) { sanitize_options_(); }

    /// Integrate one depth map.
    ///
    /// @param depth_m     Single-channel metric depth in metres (CV_32F; CV_64F
    ///                    is accepted). Non-positive/NaN pixels are skipped.
    /// @param color_bgr   Optional colour image (CV_8U BGR/BGRA, or floating
    ///                    point in [0,1] or [0,255]). Resized to the depth size.
    /// @param pose_w2c    Row-major 3x4 world-to-camera pose:
    ///                    r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz.
    /// @param k           Pinhole intrinsics in pixels for the depth resolution.
    /// @returns true when the frame was integrated and appended to the
    ///          trajectory; false for malformed input or a backend failure.
    [[nodiscard]] bool integrate(const cv::Mat &depth_m, const cv::Mat &color_bgr,
                                 const std::array<float, 12> &pose_w2c, const Pinhole &k) {
        if (depth_m.empty() || depth_m.channels() != 1) {
            return false;
        }
        if (!(k.fx > 0.0F) || !(k.fy > 0.0F)) {
            return false;
        }
        if (!pose_is_finite(pose_w2c)) {
            return false;
        }

        // w2c -> c2w exactly once, before touching either backend.
        const std::array<float, 12> pose_c2w = invert_pose_w2c_to_c2w(pose_w2c);
        if (!pose_is_finite(pose_c2w)) {
            return false;
        }

        cv::Mat depth;
        if (depth_m.type() == CV_32FC1) {
            depth = depth_m;
        } else if (depth_m.type() == CV_64FC1) {
            depth_m.convertTo(depth, CV_32F);
        } else {
            return false;
        }

#ifdef DA3_FUSION_USE_OPEN3D
        if (ensure_open3d_()) {
            const cv::Mat color = prepare_color_(color_bgr, depth.rows, depth.cols);
            if (integrate_open3d_(depth, color, pose_c2w, k)) {
                trajectory_.push_back(pose_c2w);
                return true;
            }
            // Open3D failed mid-run: degrade permanently to the fallback so the
            // accumulated points and trajectory stay consistent.
            open3d_ready_ = false;
            open3d_failed_ = true;
        }
#endif

        if (!accumulate_fallback_(depth, color_bgr, pose_c2w, k)) {
            return false;
        }
        trajectory_.push_back(pose_c2w);
        return true;
    }

    /// Export the fused surface.
    ///
    /// With Open3D this writes the extracted triangle mesh (falling back to the
    /// extracted point cloud when the TSDF has no surface). Without Open3D, the
    /// accumulated back-projected points are written as ASCII PLY.
    ///
    /// @returns true when a file was written; false when there is no geometry
    ///          yet or the file could not be opened.
    [[nodiscard]] bool extract_mesh(const std::filesystem::path &fused_mesh_ply) {
#ifdef DA3_FUSION_USE_OPEN3D
        if (open3d_ready_ && model_) {
            try {
                auto mesh = model_->ExtractTriangleMesh();
                if (mesh && mesh->GetVertices().GetLength() > 0) {
                    if (ensure_parent_directories_(fused_mesh_ply) &&
                        open3d::t::io::WriteTriangleMesh(fused_mesh_ply.string(), *mesh,
                                                         /*write_ascii=*/true,
                                                         /*compressed=*/false)) {
                        return true;
                    }
                    return false;
                }
                auto cloud = model_->ExtractPointCloud();
                if (cloud && cloud->GetPoints().GetLength() > 0) {
                    if (ensure_parent_directories_(fused_mesh_ply) &&
                        open3d::t::io::WritePointCloud(fused_mesh_ply.string(), *cloud,
                                                       /*write_ascii=*/true,
                                                       /*compressed=*/false)) {
                        return true;
                    }
                    return false;
                }
            } catch (...) {
                return false;
            }
        }
#endif
        if (points_.empty()) {
            return false;
        }
        return write_fallback_ply_(fused_mesh_ply);
    }

    /// Camera-to-world poses accumulated by successful `integrate()` calls, in
    /// input order. Each entry is a row-major 3x4 c2w matrix.
    [[nodiscard]] std::vector<std::array<float, 12>> trajectory_c2w() const { return trajectory_; }

    /// Number of successfully integrated frames.
    [[nodiscard]] std::size_t frame_count() const noexcept { return trajectory_.size(); }

    /// Whether the Open3D tensor SLAM backend is active for this instance.
    [[nodiscard]] bool is_open3d_backend() const noexcept {
#ifdef DA3_FUSION_USE_OPEN3D
        return open3d_ready_;
#else
        return false;
#endif
    }

  private:
    /// One fallback point with its colour already in RGB order.
    struct Point {
        float x{0.0F};
        float y{0.0F};
        float z{0.0F};
        std::uint8_t r{255};
        std::uint8_t g{255};
        std::uint8_t b{255};
    };

    static bool pose_is_finite(const std::array<float, 12> &pose) noexcept {
        return std::ranges::all_of(pose, [](float v) { return std::isfinite(v); });
    }

    void sanitize_options_() {
        if (!(options_.voxel_length > 0.0F)) {
            options_.voxel_length = 0.01F;
        }
        if (!(options_.sdf_trunc > options_.voxel_length)) {
            options_.sdf_trunc = 4.0F * options_.voxel_length;
        }
        if (options_.block_count <= 0) {
            options_.block_count = 40000;
        }
        if (options_.stride < 1) {
            options_.stride = 1;
        }
    }

    [[nodiscard]] static bool ensure_parent_directories_(const std::filesystem::path &path) {
        const std::filesystem::path parent = path.parent_path();
        if (parent.empty()) {
            return true;
        }
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        return !ec;
    }

    /// Convert an arbitrary colour image to 8-bit BGR at the depth resolution.
    [[nodiscard]] static cv::Mat prepare_color_(const cv::Mat &color_bgr, int rows, int cols) {
        if (color_bgr.empty() || rows <= 0 || cols <= 0) {
            return {};
        }
        cv::Mat bgr;
        if (color_bgr.channels() == 4) {
            cv::cvtColor(color_bgr, bgr, cv::COLOR_BGRA2BGR);
        } else if (color_bgr.channels() == 1) {
            cv::cvtColor(color_bgr, bgr, cv::COLOR_GRAY2BGR);
        } else {
            bgr = color_bgr;
        }
        if (bgr.depth() != CV_8U) {
            cv::Mat flat = bgr.reshape(1);
            double lo = 0.0;
            double hi = 0.0;
            cv::minMaxLoc(flat, &lo, &hi);
            const double scale = hi <= 1.5 ? 255.0 : 1.0;
            cv::Mat converted;
            bgr.convertTo(converted, CV_8U, scale);
            bgr = converted;
        }
        if (bgr.cols != cols || bgr.rows != rows) {
            cv::Mat resized;
            cv::resize(bgr, resized, cv::Size(cols, rows), 0.0, 0.0, cv::INTER_LINEAR);
            bgr = resized;
        }
        return bgr;
    }

    /// Back-project valid depth pixels with the c2w pose and accumulate them.
    bool accumulate_fallback_(const cv::Mat &depth, const cv::Mat &color_bgr, const std::array<float, 12> &pose_c2w,
                              const Pinhole &k) {
        const cv::Mat color = prepare_color_(color_bgr, depth.rows, depth.cols);
        const int step = options_.stride > 0 ? options_.stride : 1;
        const float max_depth = options_.max_depth_m > 0.0F ? options_.max_depth_m : std::numeric_limits<float>::max();
        const float fx = k.fx;
        const float fy = k.fy;
        const float cx = k.cx;
        const float cy = k.cy;

        for (int v = 0; v < depth.rows; v += step) {
            const float *depth_row = depth.ptr<float>(v);
            for (int u = 0; u < depth.cols; u += step) {
                const float z = depth_row[u];
                if (!(z > 0.0F) || !std::isfinite(z) || z > max_depth) {
                    continue;
                }
                if (options_.max_points > 0 && points_.size() >= options_.max_points) {
                    return true;
                }

                const float cam_x = (static_cast<float>(u) - cx) * z / fx;
                const float cam_y = (static_cast<float>(v) - cy) * z / fy;

                Point point;
                point.x = pose_c2w[0] * cam_x + pose_c2w[1] * cam_y + pose_c2w[2] * z + pose_c2w[3];
                point.y = pose_c2w[4] * cam_x + pose_c2w[5] * cam_y + pose_c2w[6] * z + pose_c2w[7];
                point.z = pose_c2w[8] * cam_x + pose_c2w[9] * cam_y + pose_c2w[10] * z + pose_c2w[11];
                if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                    continue;
                }

                if (!color.empty()) {
                    const cv::Vec3b &bgr = color.at<cv::Vec3b>(v, u);
                    point.b = bgr[0];
                    point.g = bgr[1];
                    point.r = bgr[2];
                }
                points_.push_back(point);
            }
        }
        return true;
    }

    /// ASCII PLY writer used by the fallback (and by builds without Open3D).
    [[nodiscard]] bool write_fallback_ply_(const std::filesystem::path &path) const {
        if (!ensure_parent_directories_(path)) {
            return false;
        }
        std::ofstream out(path);
        if (!out) {
            return false;
        }
        out << "ply\n"
            << "format ascii 1.0\n"
            << "comment da3::fusion::FusionIntegrator fallback point cloud\n"
            << "element vertex " << points_.size() << "\n"
            << "property float x\n"
            << "property float y\n"
            << "property float z\n"
            << "property uchar red\n"
            << "property uchar green\n"
            << "property uchar blue\n"
            << "end_header\n";
        out << std::setprecision(9);
        for (const Point &p : points_) {
            out << p.x << ' ' << p.y << ' ' << p.z << ' ' << static_cast<int>(p.r) << ' ' << static_cast<int>(p.g)
                << ' ' << static_cast<int>(p.b) << '\n';
        }
        return static_cast<bool>(out);
    }

#ifdef DA3_FUSION_USE_OPEN3D
    /// Lazily construct the Open3D SLAM model on first use.
    bool ensure_open3d_() {
        if (open3d_ready_) {
            return true;
        }
        if (open3d_failed_) {
            return false;
        }
        try {
            device_ = open3d::core::Device("CPU:0");
            model_ = std::make_shared<open3d::t::pipelines::slam::Model>(
                options_.voxel_length, options_.sdf_trunc, options_.block_count, device_, open3d::core::Float32);
            open3d_ready_ = true;
        } catch (...) {
            model_.reset();
            open3d_failed_ = true;
        }
        return open3d_ready_;
    }

    /// Call whichever single-frame integration member the installed Open3D
    /// declares. Current releases expose `ProcessFrame`; older ones expose
    /// `Integrate`. Each call is guarded by a requires-expression, so the
    /// branch naming a missing member is never instantiated and never a hard
    /// compile error.
    template <typename ModelT, typename FrameT>
    static void call_model_entry_point_(ModelT &model, const std::shared_ptr<FrameT> &frame) {
        if constexpr (requires(ModelT &m, const std::shared_ptr<FrameT> &f) { m.ProcessFrame(f); }) {
            model.ProcessFrame(frame);
        } else if constexpr (requires(ModelT &m, const std::shared_ptr<FrameT> &f) { m.Integrate(f); }) {
            model.Integrate(frame);
        } else {
            static_assert(detail::dependent_false<ModelT>,
                          "Open3D slam::Model exposes neither ProcessFrame nor Integrate");
        }
    }

    /// Feed one pre-validated frame to Open3D using an already-inverted c2w pose.
    bool integrate_open3d_(const cv::Mat &depth, const cv::Mat &color, const std::array<float, 12> &pose_c2w,
                           const Pinhole &k) {
        if (!model_ || depth.empty()) {
            return false;
        }
        try {
            const int rows = depth.rows;
            const int cols = depth.cols;

            std::vector<float> depth_data(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
            for (int v = 0; v < rows; ++v) {
                const float *row = depth.ptr<float>(v);
                std::copy(row, row + cols,
                          depth_data.begin() + static_cast<std::ptrdiff_t>(v) * static_cast<std::ptrdiff_t>(cols));
            }
            auto depth_tensor = open3d::core::Tensor::Empty({rows, cols, 1}, open3d::core::Float32, device_);
            std::memcpy(depth_tensor.GetDataPtr(), depth_data.data(), depth_data.size() * sizeof(float));

            std::vector<float> color_data;
            if (!color.empty()) {
                color_data.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * 3U);
                for (int v = 0; v < rows; ++v) {
                    const cv::Vec3b *row = color.ptr<cv::Vec3b>(v);
                    for (int u = 0; u < cols; ++u) {
                        const std::size_t base = (static_cast<std::size_t>(v) * static_cast<std::size_t>(cols) +
                                                  static_cast<std::size_t>(u)) *
                                                 3U;
                        color_data[base + 0U] = static_cast<float>(row[u][2]) / 255.0F;
                        color_data[base + 1U] = static_cast<float>(row[u][1]) / 255.0F;
                        color_data[base + 2U] = static_cast<float>(row[u][0]) / 255.0F;
                    }
                }
            }

            double intrinsics[9] = {static_cast<double>(k.fx),
                                    0.0,
                                    static_cast<double>(k.cx),
                                    0.0,
                                    static_cast<double>(k.fy),
                                    static_cast<double>(k.cy),
                                    0.0,
                                    0.0,
                                    1.0};
            auto intrinsics_tensor = open3d::core::Tensor::Empty({3, 3}, open3d::core::Float64, device_);
            std::memcpy(intrinsics_tensor.GetDataPtr(), intrinsics, sizeof(intrinsics));

            auto frame = std::make_shared<open3d::t::pipelines::slam::Frame>(rows, cols, intrinsics_tensor, device_);
            frame->SetDepth(depth_tensor);
            if (!color_data.empty()) {
                auto color_tensor = open3d::core::Tensor::Empty({rows, cols, 3}, open3d::core::Float32, device_);
                std::memcpy(color_tensor.GetDataPtr(), color_data.data(), color_data.size() * sizeof(float));
                frame->SetColor(color_tensor);
            }

            double pose[16] = {static_cast<double>(pose_c2w[0]),
                               static_cast<double>(pose_c2w[1]),
                               static_cast<double>(pose_c2w[2]),
                               static_cast<double>(pose_c2w[3]),
                               static_cast<double>(pose_c2w[4]),
                               static_cast<double>(pose_c2w[5]),
                               static_cast<double>(pose_c2w[6]),
                               static_cast<double>(pose_c2w[7]),
                               static_cast<double>(pose_c2w[8]),
                               static_cast<double>(pose_c2w[9]),
                               static_cast<double>(pose_c2w[10]),
                               static_cast<double>(pose_c2w[11]),
                               0.0,
                               0.0,
                               0.0,
                               1.0};
            auto pose_tensor = open3d::core::Tensor::Empty({4, 4}, open3d::core::Float64, device_);
            std::memcpy(pose_tensor.GetDataPtr(), pose, sizeof(pose));
            frame->SetPose(pose_tensor);

            // Open3D renamed the single-frame entry point across releases;
            // dispatch on whichever member the installed headers declare.
            call_model_entry_point_(*model_, frame);
            return true;
        } catch (...) {
            return false;
        }
    }

    std::shared_ptr<open3d::t::pipelines::slam::Model> model_;
    open3d::core::Device device_{"CPU:0"};
    bool open3d_ready_{false};
    bool open3d_failed_{false};
#endif

    Options options_{};
    std::vector<std::array<float, 12>> trajectory_;
    std::vector<Point> points_;
};

} // namespace da3::fusion

#ifdef DA3_FUSION_USE_OPEN3D
#undef DA3_FUSION_USE_OPEN3D
#endif
