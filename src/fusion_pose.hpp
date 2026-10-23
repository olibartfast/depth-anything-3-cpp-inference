#pragma once

// Pose providers for the fusion stage.
//
// Pose contract
// -------------
// Every provider emits a world-to-camera (w2c) pose as 12 row-major floats:
//
//   r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
//
// with p_cam = R * p_world + t. No provider inverts a pose; inversion stays in
// the integrator (`invert_pose_w2c_to_c2w` in fusion_utils.hpp), matching the
// pose convention in poses.txt. This header is OpenCV + STL only and adds no
// Open3D dependency to the default build.

#include "depth_anything3_inference.hpp"
#include "fusion_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace da3::fusion {

/// Sequential source of w2c poses for the fusion runner.
///
/// `next()` returns false when the source is exhausted or cannot produce a
/// trustworthy pose. `has_real_tracking()` reports whether the source is
/// real tracking rather than a model fallback, so callers can distinguish a
/// tracked trajectory from identity-filled model output.
class PoseProvider {
  public:
    virtual ~PoseProvider() = default;
    PoseProvider(const PoseProvider &) = delete;
    PoseProvider &operator=(const PoseProvider &) = delete;
    PoseProvider(PoseProvider &&) = delete;
    PoseProvider &operator=(PoseProvider &&) = delete;

    /// Write the next w2c pose and advance. Returns false on exhaustion or
    /// failure; `pose_w2c` is left untouched in that case.
    [[nodiscard]] virtual bool next(std::array<float, 12> &pose_w2c) = 0;

    /// True when the poses come from real tracking (external file or ICP),
    /// false when the model had no pose head and may only offer identity.
    [[nodiscard]] virtual bool has_real_tracking() const = 0;

  protected:
    PoseProvider() = default;
};

/// Poses read from a `poses.txt`-style list (or supplied directly by a caller).
///
/// The constructor taking a path uses `read_poses_txt()`; an unreadable or
/// empty file leaves the provider exhausted so `next()` returns false.
class ExternalPoseProvider final : public PoseProvider {
  public:
    explicit ExternalPoseProvider(std::vector<std::array<float, 12>> poses) : poses_(std::move(poses)) {}

    explicit ExternalPoseProvider(const std::filesystem::path &path) : poses_(load(path)) {}

    [[nodiscard]] bool next(std::array<float, 12> &pose_w2c) override {
        if (index_ >= poses_.size()) {
            return false;
        }
        pose_w2c = poses_[index_];
        ++index_;
        return true;
    }

    [[nodiscard]] bool has_real_tracking() const override { return true; }

  private:
    [[nodiscard]] static std::vector<std::array<float, 12>> load(const std::filesystem::path &path) {
        std::vector<std::array<float, 12>> poses;
        if (!read_poses_txt(path, poses)) {
            poses.clear();
        }
        return poses;
    }

    std::vector<std::array<float, 12>> poses_;
    std::size_t index_{0};
};

/// Poses accumulated from the DA3 pose head (`tasks::FrameData::pose`).
///
/// A frame whose `pose_from_model` flag is false is never emitted: `next()`
/// returns false instead of handing back the identity fallback, and
/// `has_real_tracking()` is true only when every frame carries a model pose.
class ModelPoseProvider final : public PoseProvider {
  public:
    explicit ModelPoseProvider(std::vector<::tasks::FrameData> frames) : frames_(std::move(frames)) {}

    [[nodiscard]] bool next(std::array<float, 12> &pose_w2c) override {
        if (index_ >= frames_.size() || !frames_[index_].pose_from_model) {
            return false;
        }
        pose_w2c = frames_[index_].pose;
        ++index_;
        return true;
    }

    [[nodiscard]] bool has_real_tracking() const override {
        if (frames_.empty()) {
            return false;
        }
        return std::all_of(frames_.begin(), frames_.end(),
                           [](const ::tasks::FrameData &frame) { return frame.pose_from_model; });
    }

  private:
    std::vector<::tasks::FrameData> frames_;
    std::size_t index_{0};
};

/// Tuning for IcpPoseProvider. Distances are metres, angles radians (degrees
/// only where the field name says so).
struct IcpOptions {
    /// Requested coarse-to-fine levels. Levels whose image would fall below
    /// 16 px on a side are skipped, so at least one level always runs.
    int pyramid_levels{3};
    /// Maximum Gauss-Newton iterations per level.
    int max_iterations{30};

    /// Evaluation window, matching the fusion metrics.
    float min_depth_m{0.3F};
    float max_depth_m{3.0F};

    /// Huber threshold on the point-to-plane residual (metres).
    float huber_delta_m{0.01F};
    /// Correspondence distance gate (metres) applied before Huber weighting.
    float max_correspondence_m{0.10F};
    /// Maximum angle between the current and reference surface normals.
    float max_normal_angle_deg{45.0F};

    /// Per-level minimum inlier count and inlier ratio (inliers / valid
    /// current pixels). Below either, the level (and the frame) fails.
    std::size_t min_inliers{30};
    float min_inlier_ratio{0.1F};

    /// Final acceptance gate on the mean absolute inlier residual (metres).
    float max_mean_residual_m{0.02F};

    /// A level has converged when the accepted twist increment is below both.
    float convergence_rotation_rad{1e-4F};
    float convergence_translation_m{1e-5F};
};

namespace detail {

using Vec3d = std::array<double, 3>;
using Mat3d = std::array<double, 9>;

/// Rigid transform p_cam = R * p_prev + t, stored row-major.
struct Pose3d {
    Mat3d rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    Vec3d translation{0.0, 0.0, 0.0};
};

[[nodiscard]] inline Vec3d mat_vec(const Mat3d &m, const Vec3d &v) noexcept {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

[[nodiscard]] inline Vec3d mat_t_vec(const Mat3d &m, const Vec3d &v) noexcept {
    return {m[0] * v[0] + m[3] * v[1] + m[6] * v[2], m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
            m[2] * v[0] + m[5] * v[1] + m[8] * v[2]};
}

[[nodiscard]] inline Mat3d mat_mul(const Mat3d &a, const Mat3d &b) noexcept {
    return {
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8], a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    };
}

/// Compose a * b, i.e. apply b then a.
[[nodiscard]] inline Pose3d compose(const Pose3d &a, const Pose3d &b) noexcept {
    const Vec3d rotated = mat_vec(a.rotation, b.translation);
    return {mat_mul(a.rotation, b.rotation),
            Vec3d{rotated[0] + a.translation[0], rotated[1] + a.translation[1], rotated[2] + a.translation[2]}};
}

[[nodiscard]] inline Pose3d pose_from_array(const std::array<float, 12> &pose) noexcept {
    return {{static_cast<double>(pose[0]), static_cast<double>(pose[1]), static_cast<double>(pose[2]),
             static_cast<double>(pose[4]), static_cast<double>(pose[5]), static_cast<double>(pose[6]),
             static_cast<double>(pose[8]), static_cast<double>(pose[9]), static_cast<double>(pose[10])},
            {static_cast<double>(pose[3]), static_cast<double>(pose[7]), static_cast<double>(pose[11])}};
}

[[nodiscard]] inline std::array<float, 12> pose_to_array(const Pose3d &pose) noexcept {
    return {static_cast<float>(pose.rotation[0]), static_cast<float>(pose.rotation[1]),
            static_cast<float>(pose.rotation[2]), static_cast<float>(pose.translation[0]),
            static_cast<float>(pose.rotation[3]), static_cast<float>(pose.rotation[4]),
            static_cast<float>(pose.rotation[5]), static_cast<float>(pose.translation[1]),
            static_cast<float>(pose.rotation[6]), static_cast<float>(pose.rotation[7]),
            static_cast<float>(pose.rotation[8]), static_cast<float>(pose.translation[2])};
}

/// SE(3) exponential map for xi = (omega, nu).
[[nodiscard]] inline Pose3d exp_twist(const std::array<double, 6> &xi) noexcept {
    const double wx = xi[0];
    const double wy = xi[1];
    const double wz = xi[2];
    const double theta2 = wx * wx + wy * wy + wz * wz;
    const double theta = std::sqrt(theta2);

    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    if (theta < 1e-8) {
        // Taylor expansions of sin(t)/t, (1-cos t)/t^2 and (t-sin t)/t^3.
        a = 1.0 - theta2 / 6.0;
        b = 0.5 - theta2 / 24.0;
        c = 1.0 / 6.0 - theta2 / 120.0;
    } else {
        const double sin_theta = std::sin(theta);
        const double cos_theta = std::cos(theta);
        a = sin_theta / theta;
        b = (1.0 - cos_theta) / theta2;
        c = (theta - sin_theta) / (theta2 * theta);
    }

    const Mat3d omega{0.0, -wz, wy, wz, 0.0, -wx, -wy, wx, 0.0};
    const Mat3d omega2 = mat_mul(omega, omega);

    Mat3d rotation{};
    Mat3d v_mat{};
    for (std::size_t i = 0; i < 9; ++i) {
        const double identity = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;
        rotation[i] = identity + a * omega[i] + b * omega2[i];
        v_mat[i] = identity + b * omega[i] + c * omega2[i];
    }
    return {rotation, mat_vec(v_mat, Vec3d{xi[3], xi[4], xi[5]})};
}

/// Dense 6x6 solve with partial pivoting. Returns false on a singular system.
[[nodiscard]] inline bool solve6(std::array<std::array<double, 6>, 6> matrix, std::array<double, 6> rhs,
                                 std::array<double, 6> &solution) noexcept {
    for (std::size_t col = 0; col < 6; ++col) {
        std::size_t pivot = col;
        double best = std::abs(matrix[col][col]);
        for (std::size_t row = col + 1; row < 6; ++row) {
            const double candidate = std::abs(matrix[row][col]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (!(best > 1e-12)) {
            return false;
        }
        if (pivot != col) {
            std::swap(matrix[pivot], matrix[col]);
            std::swap(rhs[pivot], rhs[col]);
        }
        const double diag = matrix[col][col];
        for (std::size_t row = col + 1; row < 6; ++row) {
            const double factor = matrix[row][col] / diag;
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t k = col; k < 6; ++k) {
                matrix[row][k] -= factor * matrix[col][k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    for (std::size_t i = 6; i-- > 0;) {
        double sum = rhs[i];
        for (std::size_t j = i + 1; j < 6; ++j) {
            sum -= matrix[i][j] * solution[j];
        }
        solution[i] = sum / matrix[i][i];
        if (!std::isfinite(solution[i])) {
            return false;
        }
    }
    return true;
}

/// Vertex and normal maps for one pyramid level, in that level's camera frame.
struct DepthMaps {
    cv::Mat vertices; // CV_32FC3, NaN when the depth is outside the window
    cv::Mat normals;  // CV_32FC3, NaN where a normal could not be estimated
    Pinhole intrinsics;
};

[[nodiscard]] inline bool is_valid_depth(float depth, float min_depth, float max_depth) noexcept {
    return std::isfinite(depth) && depth >= min_depth && depth <= max_depth;
}

[[nodiscard]] inline bool is_finite_vec(const cv::Vec3f &value) noexcept {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

/// Decimate by two, taking the top-left sample. Pixel (u, v) of the result
/// therefore corresponds exactly to pixel (2u, 2v) of the input, which keeps
/// the scaled intrinsics (fx/2, cx/2) consistent without a half-pixel shift.
[[nodiscard]] inline cv::Mat downsample_depth(const cv::Mat &depth) {
    const int rows = depth.rows / 2;
    const int cols = depth.cols / 2;
    cv::Mat out(rows, cols, CV_32F, cv::Scalar(0.0));
    for (int y = 0; y < rows; ++y) {
        const float *src = depth.ptr<float>(2 * y);
        float *dst = out.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            dst[x] = src[2 * x];
        }
    }
    return out;
}

/// Back-project a depth map and estimate normals from central differences.
[[nodiscard]] inline DepthMaps build_depth_maps(const cv::Mat &depth, const Pinhole &intrinsics, float min_depth,
                                                float max_depth) {
    CV_Assert(depth.type() == CV_32F);

    DepthMaps maps;
    maps.intrinsics = intrinsics;
    maps.vertices = cv::Mat(depth.rows, depth.cols, CV_32FC3);
    maps.normals = cv::Mat(depth.rows, depth.cols, CV_32FC3);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const cv::Scalar invalid = cv::Scalar::all(static_cast<double>(nan));
    maps.vertices.setTo(invalid);
    maps.normals.setTo(invalid);

    const float fx = intrinsics.fx > 0.0F ? intrinsics.fx : 1.0F;
    const float fy = intrinsics.fy > 0.0F ? intrinsics.fy : 1.0F;

    for (int y = 0; y < depth.rows; ++y) {
        const float *depth_row = depth.ptr<float>(y);
        cv::Vec3f *vertex_row = maps.vertices.ptr<cv::Vec3f>(y);
        for (int x = 0; x < depth.cols; ++x) {
            const float z = depth_row[x];
            if (!is_valid_depth(z, min_depth, max_depth)) {
                continue;
            }
            const float cam_x = (static_cast<float>(x) - intrinsics.cx) * z / fx;
            const float cam_y = (static_cast<float>(y) - intrinsics.cy) * z / fy;
            vertex_row[x] = cv::Vec3f(cam_x, cam_y, z);
        }
    }

    for (int y = 1; y < depth.rows - 1; ++y) {
        const cv::Vec3f *up = maps.vertices.ptr<cv::Vec3f>(y - 1);
        const cv::Vec3f *mid = maps.vertices.ptr<cv::Vec3f>(y);
        const cv::Vec3f *down = maps.vertices.ptr<cv::Vec3f>(y + 1);
        cv::Vec3f *normal_row = maps.normals.ptr<cv::Vec3f>(y);
        for (int x = 1; x < depth.cols - 1; ++x) {
            if (!is_finite_vec(mid[x - 1]) || !is_finite_vec(mid[x + 1]) || !is_finite_vec(up[x]) ||
                !is_finite_vec(down[x])) {
                continue;
            }
            const cv::Vec3f dx = mid[x + 1] - mid[x - 1];
            const cv::Vec3f dy = down[x] - up[x];
            cv::Vec3f normal = dx.cross(dy);
            const float norm = std::sqrt(normal.dot(normal));
            if (!(norm > 1e-6F)) {
                continue;
            }
            normal_row[x] = normal / norm;
        }
    }
    return maps;
}

/// Bilinear lookup of a vertex and normal, requiring all four corners valid.
[[nodiscard]] inline bool interpolate_maps(const DepthMaps &maps, double u, double v, Vec3d &vertex,
                                           Vec3d &normal) noexcept {
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0.0 || v < 0.0) {
        return false;
    }
    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= maps.vertices.cols || y0 + 1 >= maps.vertices.rows) {
        return false;
    }

    const cv::Vec3f &v00 = maps.vertices.at<cv::Vec3f>(y0, x0);
    const cv::Vec3f &v10 = maps.vertices.at<cv::Vec3f>(y0, x0 + 1);
    const cv::Vec3f &v01 = maps.vertices.at<cv::Vec3f>(y0 + 1, x0);
    const cv::Vec3f &v11 = maps.vertices.at<cv::Vec3f>(y0 + 1, x0 + 1);
    const cv::Vec3f &n00 = maps.normals.at<cv::Vec3f>(y0, x0);
    const cv::Vec3f &n10 = maps.normals.at<cv::Vec3f>(y0, x0 + 1);
    const cv::Vec3f &n01 = maps.normals.at<cv::Vec3f>(y0 + 1, x0);
    const cv::Vec3f &n11 = maps.normals.at<cv::Vec3f>(y0 + 1, x0 + 1);
    if (!is_finite_vec(v00) || !is_finite_vec(v10) || !is_finite_vec(v01) || !is_finite_vec(v11) ||
        !is_finite_vec(n00) || !is_finite_vec(n10) || !is_finite_vec(n01) || !is_finite_vec(n11)) {
        return false;
    }

    const double tx = u - static_cast<double>(x0);
    const double ty = v - static_cast<double>(y0);
    const double w00 = (1.0 - tx) * (1.0 - ty);
    const double w10 = tx * (1.0 - ty);
    const double w01 = (1.0 - tx) * ty;
    const double w11 = tx * ty;

    vertex = {w00 * static_cast<double>(v00[0]) + w10 * static_cast<double>(v10[0]) +
                  w01 * static_cast<double>(v01[0]) + w11 * static_cast<double>(v11[0]),
              w00 * static_cast<double>(v00[1]) + w10 * static_cast<double>(v10[1]) +
                  w01 * static_cast<double>(v01[1]) + w11 * static_cast<double>(v11[1]),
              w00 * static_cast<double>(v00[2]) + w10 * static_cast<double>(v10[2]) +
                  w01 * static_cast<double>(v01[2]) + w11 * static_cast<double>(v11[2])};
    normal = {w00 * static_cast<double>(n00[0]) + w10 * static_cast<double>(n10[0]) +
                  w01 * static_cast<double>(n01[0]) + w11 * static_cast<double>(n11[0]),
              w00 * static_cast<double>(n00[1]) + w10 * static_cast<double>(n10[1]) +
                  w01 * static_cast<double>(n01[1]) + w11 * static_cast<double>(n11[1]),
              w00 * static_cast<double>(n00[2]) + w10 * static_cast<double>(n10[2]) +
                  w01 * static_cast<double>(n01[2]) + w11 * static_cast<double>(n11[2])};

    const double norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (!(norm > 1e-12)) {
        return false;
    }
    normal = {normal[0] / norm, normal[1] / norm, normal[2] / norm};
    return true;
}

struct Linearization {
    std::array<std::array<double, 6>, 6> h{};
    std::array<double, 6> b{};
    std::size_t inliers{0};
    std::size_t valid_points{0};
    double residual_sum{0.0};
};

/// One point-to-plane linearisation at a fixed relative pose.
///
/// Residual: r = n_ref . (T^-1 q - v_ref), where q is the current vertex.
/// With a left perturbation T <- exp(xi) T the Jacobian is
///   d r / d omega = (R n_ref) x q,   d r / d nu = -(R n_ref).
[[nodiscard]] inline Linearization linearize(const DepthMaps &reference, const DepthMaps &current,
                                             const Pose3d &relative, const IcpOptions &options) {
    constexpr double kPi = 3.14159265358979323846;
    const double cos_limit = std::cos(static_cast<double>(options.max_normal_angle_deg) * (kPi / 180.0));
    const double min_depth = static_cast<double>(options.min_depth_m);
    const double max_depth = static_cast<double>(options.max_depth_m);
    const double max_correspondence = static_cast<double>(options.max_correspondence_m);
    const double huber_delta = static_cast<double>(options.huber_delta_m);

    Linearization result;
    if (!(max_correspondence > 0.0) || !(huber_delta > 0.0)) {
        return result;
    }

    const double fx = static_cast<double>(reference.intrinsics.fx);
    const double fy = static_cast<double>(reference.intrinsics.fy);
    const double cx = static_cast<double>(reference.intrinsics.cx);
    const double cy = static_cast<double>(reference.intrinsics.cy);

    for (int y = 1; y < current.vertices.rows - 1; ++y) {
        const cv::Vec3f *vertex_row = current.vertices.ptr<cv::Vec3f>(y);
        const cv::Vec3f *normal_row = current.normals.ptr<cv::Vec3f>(y);
        for (int x = 1; x < current.vertices.cols - 1; ++x) {
            const cv::Vec3f &current_vertex = vertex_row[x];
            const cv::Vec3f &current_normal = normal_row[x];
            if (!is_finite_vec(current_vertex)) {
                continue;
            }
            ++result.valid_points;
            if (!is_finite_vec(current_normal)) {
                continue;
            }

            const Vec3d q{static_cast<double>(current_vertex[0]), static_cast<double>(current_vertex[1]),
                          static_cast<double>(current_vertex[2])};
            const Vec3d shifted{q[0] - relative.translation[0], q[1] - relative.translation[1],
                                q[2] - relative.translation[2]};
            const Vec3d p = mat_t_vec(relative.rotation, shifted);
            if (p[2] < min_depth || p[2] > max_depth) {
                continue;
            }

            const double u = fx * p[0] / p[2] + cx;
            const double v = fy * p[1] / p[2] + cy;
            Vec3d reference_vertex{};
            Vec3d reference_normal{};
            if (!interpolate_maps(reference, u, v, reference_vertex, reference_normal)) {
                continue;
            }

            const Vec3d current_normal_prev{static_cast<double>(current_normal[0]),
                                            static_cast<double>(current_normal[1]),
                                            static_cast<double>(current_normal[2])};
            const Vec3d current_normal_in_ref = mat_t_vec(relative.rotation, current_normal_prev);
            const double normal_dot = current_normal_in_ref[0] * reference_normal[0] +
                                      current_normal_in_ref[1] * reference_normal[1] +
                                      current_normal_in_ref[2] * reference_normal[2];
            if (std::abs(normal_dot) < cos_limit) {
                continue;
            }

            const double residual = reference_normal[0] * (p[0] - reference_vertex[0]) +
                                    reference_normal[1] * (p[1] - reference_vertex[1]) +
                                    reference_normal[2] * (p[2] - reference_vertex[2]);
            if (!std::isfinite(residual) || std::abs(residual) > max_correspondence) {
                continue;
            }

            const double abs_residual = std::abs(residual);
            const double weight = abs_residual <= huber_delta ? 1.0 : huber_delta / abs_residual;
            ++result.inliers;
            result.residual_sum += abs_residual;

            const Vec3d normal_in_current = mat_vec(relative.rotation, reference_normal);
            const Vec3d jacobian_rotation{normal_in_current[1] * q[2] - normal_in_current[2] * q[1],
                                          normal_in_current[2] * q[0] - normal_in_current[0] * q[2],
                                          normal_in_current[0] * q[1] - normal_in_current[1] * q[0]};
            const double jacobian[6] = {jacobian_rotation[0],  jacobian_rotation[1],  jacobian_rotation[2],
                                        -normal_in_current[0], -normal_in_current[1], -normal_in_current[2]};
            for (std::size_t i = 0; i < 6; ++i) {
                for (std::size_t k = 0; k < 6; ++k) {
                    result.h[i][k] += weight * jacobian[i] * jacobian[k];
                }
                result.b[i] += weight * jacobian[i] * residual;
            }
        }
    }
    return result;
}

/// Align the current depth map to the reference map. Returns the relative
/// transform prev-camera -> current-camera, or nullopt when the frame cannot
/// be trusted.
///
/// The reference is the previous depth frame (frame-to-frame), not a raycast
/// TSDF model. This is the documented F2 limitation; a raycast reference can
/// replace `reference` later without changing the interface.
[[nodiscard]] inline std::optional<Pose3d> align_depth_maps(const cv::Mat &reference_depth,
                                                            const cv::Mat &current_depth, const Pinhole &intrinsics,
                                                            const IcpOptions &options) {
    if (reference_depth.empty() || current_depth.empty() || reference_depth.type() != CV_32F ||
        current_depth.type() != CV_32F) {
        return std::nullopt;
    }
    if (reference_depth.size() != current_depth.size()) {
        return std::nullopt;
    }
    if (!(intrinsics.fx > 0.0F) || !(intrinsics.fy > 0.0F)) {
        return std::nullopt;
    }
    if (!(options.min_depth_m > 0.0F) || !(options.max_depth_m > options.min_depth_m)) {
        return std::nullopt;
    }

    const int requested_levels = std::max(1, options.pyramid_levels);
    int levels = 1;
    while (levels < requested_levels && (current_depth.cols / (1 << levels)) >= 16 &&
           (current_depth.rows / (1 << levels)) >= 16) {
        ++levels;
    }

    std::vector<cv::Mat> reference_pyramid;
    std::vector<cv::Mat> current_pyramid;
    reference_pyramid.push_back(reference_depth);
    current_pyramid.push_back(current_depth);
    for (int level = 1; level < levels; ++level) {
        reference_pyramid.push_back(downsample_depth(reference_pyramid.back()));
        current_pyramid.push_back(downsample_depth(current_pyramid.back()));
    }

    std::vector<DepthMaps> reference_maps;
    std::vector<DepthMaps> current_maps;
    for (int level = 0; level < levels; ++level) {
        const float scale = static_cast<float>(1 << level);
        const Pinhole scaled{intrinsics.fx / scale, intrinsics.fy / scale, intrinsics.cx / scale,
                             intrinsics.cy / scale};
        reference_maps.push_back(build_depth_maps(reference_pyramid[static_cast<std::size_t>(level)], scaled,
                                                  options.min_depth_m, options.max_depth_m));
        current_maps.push_back(build_depth_maps(current_pyramid[static_cast<std::size_t>(level)], scaled,
                                                options.min_depth_m, options.max_depth_m));
    }

    const int max_iterations = std::max(1, options.max_iterations);
    const double min_inliers = static_cast<double>(options.min_inliers);
    const double min_ratio = static_cast<double>(options.min_inlier_ratio);
    const double convergence_rotation = static_cast<double>(options.convergence_rotation_rad);
    const double convergence_translation = static_cast<double>(options.convergence_translation_m);
    constexpr double kMaxRotationStep = 0.2;     // radians per iteration
    constexpr double kMaxTranslationStep = 0.05; // metres per iteration

    Pose3d relative;
    bool converged = false;

    for (int level = levels - 1; level >= 0; --level) {
        const DepthMaps &reference = reference_maps[static_cast<std::size_t>(level)];
        const DepthMaps &current = current_maps[static_cast<std::size_t>(level)];
        converged = false;

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            const Linearization system = linearize(reference, current, relative, options);
            if (static_cast<double>(system.inliers) < min_inliers ||
                static_cast<double>(system.inliers) < min_ratio * static_cast<double>(system.valid_points)) {
                return std::nullopt;
            }

            // Small Levenberg-style damping keeps the normal equations solvable
            // when the surface is close to degenerate.
            double trace = 0.0;
            for (std::size_t i = 0; i < 6; ++i) {
                trace += system.h[i][i];
            }
            const double damping = std::max(1e-12, 1e-6 * trace / 6.0);

            std::array<std::array<double, 6>, 6> normal_equations = system.h;
            for (std::size_t i = 0; i < 6; ++i) {
                normal_equations[i][i] += damping;
            }
            std::array<double, 6> rhs{};
            for (std::size_t i = 0; i < 6; ++i) {
                rhs[i] = -system.b[i];
            }

            std::array<double, 6> delta{};
            if (!solve6(normal_equations, rhs, delta)) {
                return std::nullopt;
            }

            double rotation_norm = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
            double translation_norm = std::sqrt(delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);
            double step_scale = 1.0;
            if (rotation_norm > kMaxRotationStep) {
                step_scale = std::min(step_scale, kMaxRotationStep / rotation_norm);
            }
            if (translation_norm > kMaxTranslationStep) {
                step_scale = std::min(step_scale, kMaxTranslationStep / translation_norm);
            }
            for (double &value : delta) {
                value *= step_scale;
            }
            rotation_norm *= step_scale;
            translation_norm *= step_scale;

            relative = compose(exp_twist(delta), relative);
            if (rotation_norm < convergence_rotation && translation_norm < convergence_translation) {
                converged = true;
                break;
            }
        }
    }

    if (!converged) {
        return std::nullopt;
    }

    // Re-evaluate at the final pose so the acceptance gates see the pose that
    // is actually returned, not the pre-update linearisation.
    const Linearization final_system = linearize(reference_maps.front(), current_maps.front(), relative, options);
    if (static_cast<double>(final_system.inliers) < min_inliers ||
        static_cast<double>(final_system.inliers) < min_ratio * static_cast<double>(final_system.valid_points)) {
        return std::nullopt;
    }
    if (final_system.inliers == 0) {
        return std::nullopt;
    }
    const double mean_residual = final_system.residual_sum / static_cast<double>(final_system.inliers);
    if (mean_residual > static_cast<double>(options.max_mean_residual_m)) {
        return std::nullopt;
    }

    return relative;
}

} // namespace detail

/// Frame-to-frame point-to-plane ICP tracker.
///
/// Each `next()` call returns the w2c pose for the next depth frame. The first
/// frame seeds the trajectory with `first_pose_w2c`; every later frame is
/// aligned to the previous frame's vertex/normal map (projective data
/// association through the current pose estimate, Huber weighting, distance and
/// normal-angle rejection, Gauss-Newton on a 6-DoF twist, three pyramid levels).
///
/// A frame is rejected (false, no pose written) when the residuals cannot be
/// trusted: too few inliers, too low an inlier ratio, a mean residual above
/// the gate, a singular normal equation, or no convergence on the finest
/// level. The provider latches that failure so a caller can never continue on
/// a stale pose. Depth must be CV_32F metric depth, same size for all frames,
/// and is used in the 0.3-3.0 m window.
///
/// Limitations (documented for F2):
///   * the reference is the previous depth frame, not a raycast/TSDF model;
///   * there is no constant-velocity motion model, so the first guess is
///     identity and very fast inter-frame motion can be rejected safely;
///   * a single instance is not internally synchronized.
class IcpPoseProvider final : public PoseProvider {
  public:
    using Options = IcpOptions;

    IcpPoseProvider(std::vector<cv::Mat> depth_frames, Pinhole intrinsics, std::array<float, 12> first_pose_w2c,
                    Options options = {})
        : depth_frames_(std::move(depth_frames)), intrinsics_(intrinsics), first_pose_w2c_(first_pose_w2c),
          options_(options) {}

    [[nodiscard]] bool next(std::array<float, 12> &pose_w2c) override {
        if (failed_ || depth_frames_.empty()) {
            return false;
        }

        if (!started_) {
            const cv::Mat &first = depth_frames_.front();
            if (first.empty() || first.type() != CV_32F) {
                failed_ = true;
                return false;
            }
            pose_w2c = first_pose_w2c_;
            previous_pose_ = first_pose_w2c_;
            previous_depth_ = first;
            index_ = 1;
            started_ = true;
            return true;
        }

        if (index_ >= depth_frames_.size()) {
            return false;
        }
        const cv::Mat &current_depth = depth_frames_[index_];
        if (current_depth.empty() || current_depth.type() != CV_32F) {
            failed_ = true;
            return false;
        }

        const std::optional<detail::Pose3d> relative =
            detail::align_depth_maps(previous_depth_, current_depth, intrinsics_, options_);
        if (!relative) {
            failed_ = true;
            return false;
        }

        const detail::Pose3d composed = detail::compose(*relative, detail::pose_from_array(previous_pose_));
        pose_w2c = detail::pose_to_array(composed);
        previous_pose_ = pose_w2c;
        previous_depth_ = current_depth;
        ++index_;
        return true;
    }

    [[nodiscard]] bool has_real_tracking() const override { return true; }

  private:
    std::vector<cv::Mat> depth_frames_;
    Pinhole intrinsics_;
    std::array<float, 12> first_pose_w2c_{};
    Options options_;

    std::size_t index_{0};
    bool started_{false};
    bool failed_{false};
    std::array<float, 12> previous_pose_{};
    cv::Mat previous_depth_;
};

} // namespace da3::fusion
