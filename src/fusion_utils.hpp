#pragma once

// Fusion helpers for the TSDF stage (A/B/C experiments).
//
// Dependency-free: pose convention fixes, mask/scale preprocessing, and
// trajectory/depth metrics. The optional Open3D `slam::Model` wrapper stays
// out of the default build; see docs/fusion.md for the wiring sketch.
//
// Convention (upstream DA3, COLMAP-compatible): poses.txt stores world-to-
// camera (w2c) 3x4 rows. Back-projection must invert to world. Open3D
// `slam::Model` frame poses are camera-to-world (c2w), so invert there too.

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace da3::fusion {

/// Pinhole intrinsics in pixels.
struct Pinhole {
    float fx{0.0F};
    float fy{0.0F};
    float cx{0.0F};
    float cy{0.0F};
};

/// Scale intrinsics when depth resolution differs from calibration resolution.
[[nodiscard]] inline Pinhole scale_intrinsics(const Pinhole &k, int src_w, int src_h, int dst_w, int dst_h) {
    if (src_w <= 0 || src_h <= 0) {
        return k;
    }
    const float sx = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float sy = static_cast<float>(dst_h) / static_cast<float>(src_h);
    return Pinhole{k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy};
}

/// Invert a 3x4 row-major w2c pose to c2w: R_c2w = R^T, t_c2w = -R^T * t.
[[nodiscard]] inline std::array<float, 12> invert_pose_w2c_to_c2w(const std::array<float, 12> &w2c) {
    // R rows: [r11 r12 r13], [r21 r22 r23], [r31 r32 r33]; t = (tx,ty,tz).
    const float r11 = w2c[0];
    const float r12 = w2c[1];
    const float r13 = w2c[2];
    const float tx = w2c[3];
    const float r21 = w2c[4];
    const float r22 = w2c[5];
    const float r23 = w2c[6];
    const float ty = w2c[7];
    const float r31 = w2c[8];
    const float r32 = w2c[9];
    const float r33 = w2c[10];
    const float tz = w2c[11];
    // Transposed rotation, negated translated offset.
    return {r11, r21, r31, -(r11 * tx + r21 * ty + r31 * tz), r12, r22, r32, -(r12 * tx + r22 * ty + r32 * tz),
            r13, r23, r33, -(r13 * tx + r23 * ty + r33 * tz)};
}

/// Back-project pixel (u,v) with depth z (meters) to world using a w2c pose.
/// Correct inverse of p_cam = R * p_world + t.
inline void backproject_w2c_to_world(const std::array<float, 12> &pose_w2c, const Pinhole &k, int u, int v, float z,
                                     float &out_x, float &out_y, float &out_z) {
    const float fx = k.fx > 0.0F ? k.fx : 1.0F;
    const float fy = k.fy > 0.0F ? k.fy : 1.0F;
    const float cam_x = (static_cast<float>(u) - k.cx) * z / fx;
    const float cam_y = (static_cast<float>(v) - k.cy) * z / fy;
    const float cam_z = z;
    // world = R^T * (cam - t).
    const float dx = cam_x - pose_w2c[3];
    const float dy = cam_y - pose_w2c[7];
    const float dz = cam_z - pose_w2c[11];
    out_x = pose_w2c[0] * dx + pose_w2c[4] * dy + pose_w2c[8] * dz;
    out_y = pose_w2c[1] * dx + pose_w2c[5] * dy + pose_w2c[9] * dz;
    out_z = pose_w2c[2] * dx + pose_w2c[6] * dy + pose_w2c[10] * dz;
}

/// Zero out depth where confidence < thresh. Empty conf returns depth clone.
[[nodiscard]] inline cv::Mat mask_low_confidence(const cv::Mat &depth, const cv::Mat &conf, float thresh) {
    CV_Assert(depth.type() == CV_32F);
    cv::Mat out = depth.clone();
    if (conf.empty()) {
        return out;
    }
    CV_Assert(conf.rows == depth.rows && conf.cols == depth.cols);
    for (int y = 0; y < depth.rows; ++y) {
        const float *c = conf.ptr<float>(y);
        float *d = out.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x) {
            if (!(c[x] >= thresh)) {
                d[x] = 0.0F;
            }
        }
    }
    return out;
}

/// Valid-pixel mask (255) with flying pixels removed: gradient magnitude from
/// Sobel on metric depth above edge_thresh is marked invalid (0).
[[nodiscard]] inline cv::Mat compute_valid_mask(const cv::Mat &depth_m, float edge_thresh) {
    CV_Assert(depth_m.type() == CV_32F);
    cv::Mat valid(depth_m.size(), CV_8U, cv::Scalar(255));
    if (!(edge_thresh > 0.0F)) {
        // Only cull non-positive depths.
        for (int y = 0; y < depth_m.rows; ++y) {
            const float *d = depth_m.ptr<float>(y);
            auto *m = valid.ptr<std::uint8_t>(y);
            for (int x = 0; x < depth_m.cols; ++x) {
                if (!(d[x] > 0.0F)) {
                    m[x] = 0;
                }
            }
        }
        return valid;
    }
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(depth_m, gx, CV_32F, 1, 0, 3);
    cv::Sobel(depth_m, gy, CV_32F, 0, 1, 3);
    for (int y = 0; y < depth_m.rows; ++y) {
        const float *d = depth_m.ptr<float>(y);
        const float *px = gx.ptr<float>(y);
        const float *py = gy.ptr<float>(y);
        auto *m = valid.ptr<std::uint8_t>(y);
        for (int x = 0; x < depth_m.cols; ++x) {
            const float g = std::sqrt(px[x] * px[x] + py[x] * py[x]);
            if (!(d[x] > 0.0F) || g > edge_thresh) {
                m[x] = 0;
            }
        }
    }
    return valid;
}

/// Median of ref/pred over valid pixels (both > eps). Returns 1.0 when empty.
[[nodiscard]] inline double estimate_scale_median(const cv::Mat &pred_m, const cv::Mat &ref_m, const cv::Mat &valid) {
    CV_Assert(pred_m.type() == CV_32F && ref_m.type() == CV_32F);
    CV_Assert(pred_m.rows == ref_m.rows && pred_m.cols == ref_m.cols);
    std::vector<double> ratios;
    ratios.reserve(static_cast<std::size_t>(pred_m.rows * pred_m.cols) / 16);
    constexpr float kEps = 1e-6F;
    for (int y = 0; y < pred_m.rows; ++y) {
        const float *p = pred_m.ptr<float>(y);
        const float *r = ref_m.ptr<float>(y);
        const std::uint8_t *m = valid.empty() ? nullptr : valid.ptr<std::uint8_t>(y);
        for (int x = 0; x < pred_m.cols; ++x) {
            if (m != nullptr && m[x] == 0) {
                continue;
            }
            if (p[x] > kEps && r[x] > kEps) {
                ratios.push_back(static_cast<double>(r[x]) / static_cast<double>(p[x]));
            }
        }
    }
    if (ratios.empty()) {
        return 1.0;
    }
    const std::size_t mid = ratios.size() / 2;
    std::nth_element(ratios.begin(), ratios.begin() + static_cast<std::ptrdiff_t>(mid), ratios.end());
    return ratios[mid];
}

/// Read poses.txt (N x 12 w2c rows). Returns false on I/O or parse failure.
[[nodiscard]] inline bool read_poses_txt(const std::filesystem::path &path, std::vector<std::array<float, 12>> &out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    out.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::array<float, 12> pose{};
        for (float &v : pose) {
            if (!(iss >> v)) {
                return false;
            }
        }
        out.push_back(pose);
    }
    return !out.empty();
}

/// Read intrinsics.txt (N x fx fy cx cy rows). Returns false on failure.
[[nodiscard]] inline bool read_intrinsics_txt(const std::filesystem::path &path,
                                              std::vector<std::array<float, 4>> &out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    out.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::array<float, 4> k{};
        for (float &v : k) {
            if (!(iss >> v)) {
                return false;
            }
        }
        out.push_back(k);
    }
    return !out.empty();
}

/// Mean absolute relative error over valid pixels: |pred-ref|/ref.
[[nodiscard]] inline double abs_rel(const cv::Mat &pred_m, const cv::Mat &ref_m, const cv::Mat &valid,
                                    double min_depth = 0.3, double max_depth = 3.0) {
    CV_Assert(pred_m.type() == CV_32F && ref_m.type() == CV_32F);
    // Depth is CV_32F, so narrow the metric bounds once up front. The inner
    // loop then compares floats against floats and only widens the values it
    // accumulates, keeping every float->double conversion explicit.
    const float min_depth_m = static_cast<float>(min_depth);
    const float max_depth_m = static_cast<float>(max_depth);
    double sum = 0.0;
    std::size_t n = 0;
    for (int y = 0; y < pred_m.rows; ++y) {
        const float *p = pred_m.ptr<float>(y);
        const float *r = ref_m.ptr<float>(y);
        const std::uint8_t *m = valid.empty() ? nullptr : valid.ptr<std::uint8_t>(y);
        for (int x = 0; x < pred_m.cols; ++x) {
            if (m != nullptr && m[x] == 0) {
                continue;
            }
            const float ref_depth = r[x];
            if (ref_depth < min_depth_m || ref_depth > max_depth_m || !(p[x] > 0.0F)) {
                continue;
            }
            const double ref_depth_d = static_cast<double>(ref_depth);
            sum += std::abs(static_cast<double>(p[x]) - ref_depth_d) / ref_depth_d;
            ++n;
        }
    }
    return n == 0 ? 0.0 : sum / static_cast<double>(n);
}

/// RMSE of translation differences between estimated and reference c2w poses.
[[nodiscard]] inline double ate_translation_rmse(const std::vector<std::array<float, 12>> &est_c2w,
                                                 const std::vector<std::array<float, 12>> &ref_c2w) {
    if (est_c2w.empty() || est_c2w.size() != ref_c2w.size()) {
        return -1.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < est_c2w.size(); ++i) {
        const double dx = static_cast<double>(est_c2w[i][3]) - static_cast<double>(ref_c2w[i][3]);
        const double dy = static_cast<double>(est_c2w[i][7]) - static_cast<double>(ref_c2w[i][7]);
        const double dz = static_cast<double>(est_c2w[i][11]) - static_cast<double>(ref_c2w[i][11]);
        sum += dx * dx + dy * dy + dz * dz;
    }
    return std::sqrt(sum / static_cast<double>(est_c2w.size()));
}

} // namespace da3::fusion
