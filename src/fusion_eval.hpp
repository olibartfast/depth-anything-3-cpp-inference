#pragma once

// Evaluation helpers for the fusion stage.
//
// Header-only and dependency-light: OpenCV for the dense depth metrics and the
// standard library for point-set metrics. Reuses the preprocessing and metric
// primitives in fusion_utils.hpp so the fusion experiments share one definition
// of abs_rel and ATE. No Open3D and no network access.

// fusion_utils.hpp is shared with the project warning set; its abs_rel() helper
// trips GCC's -Wdouble-promotion. Keep that read-only helper's noise out of
// every translation unit that pulls in the evaluation helpers, mirroring the
// guard in fusion_run.hpp.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif
#include "fusion_utils.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

namespace da3::fusion {

/// Symmetric Chamfer distance using the L1 (Manhattan) nearest-neighbour
/// distance.
///
/// Returns `0.5 * (mean_{p in a} min_{q in b} ||p - q||_1
///               + mean_{q in b} min_{p in a} ||q - p||_1)`.
/// Returns -1.0 when either input is empty, matching the sentinel used by
/// ate_translation_rmse().
[[nodiscard]] inline double chamfer_l1(const std::vector<std::array<float, 3>> &a,
                                       const std::vector<std::array<float, 3>> &b) {
    if (a.empty() || b.empty()) {
        return -1.0;
    }

    const auto one_way = [](const std::vector<std::array<float, 3>> &from,
                            const std::vector<std::array<float, 3>> &to) {
        double sum = 0.0;
        for (const std::array<float, 3> &p : from) {
            double best = std::numeric_limits<double>::infinity();
            for (const std::array<float, 3> &q : to) {
                const double d = std::abs(static_cast<double>(p[0]) - static_cast<double>(q[0])) +
                                 std::abs(static_cast<double>(p[1]) - static_cast<double>(q[1])) +
                                 std::abs(static_cast<double>(p[2]) - static_cast<double>(q[2]));
                best = std::min(best, d);
            }
            sum += best;
        }
        return sum / static_cast<double>(from.size());
    };

    return 0.5 * (one_way(a, b) + one_way(b, a));
}

/// Depth abs_rel with the fusion evaluation range (0.3 m to 3.0 m).
///
/// Thin wrapper over fusion_utils.hpp abs_rel() so callers cannot drift on the
/// metric-depth clamping defaults. An empty `valid` mask means all pixels.
[[nodiscard]] inline double eval_abs_rel(const cv::Mat &pred_m, const cv::Mat &ref_m,
                                         const cv::Mat &valid = cv::Mat()) {
    return abs_rel(pred_m, ref_m, valid, 0.3, 3.0);
}

/// ATE translation RMSE over c2w poses. Thin wrapper over
/// ate_translation_rmse(); returns -1.0 on empty or size-mismatched input.
[[nodiscard]] inline double eval_ate_translation_rmse(const std::vector<std::array<float, 12>> &est_c2w,
                                                      const std::vector<std::array<float, 12>> &ref_c2w) {
    return ate_translation_rmse(est_c2w, ref_c2w);
}

} // namespace da3::fusion
