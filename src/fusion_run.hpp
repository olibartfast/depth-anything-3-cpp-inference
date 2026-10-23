#pragma once

// Offline fusion runner for the A/B/C experiments (docs/fusion.md).
//
// The three modes exist to separate the error sources of a DA3 + depth-camera
// capture:
//
//   * a — external depth + external `poses.txt` (D435 + ICP baseline). This is
//         the reference geometry: no scale alignment is applied because both
//         the depth and the trajectory are already in the capture metric frame.
//   * b — DA3 `FrameData::depth` + external `poses.txt`. The depth scale is
//         aligned to the external depth reference with
//         `estimate_scale_median()` before integration.
//   * c — DA3 depth + the poses tracked by the model itself (`FrameData::pose`).
//         No external pose file and no scale alignment.
//
// All dense preprocessing goes through the dependency-free helpers in
// fusion_utils.hpp and the fixed order documented there:
//
//   scale_intrinsics -> mask_low_confidence (only with --conf-thresh)
//                    -> compute_valid_mask -> estimate_scale_median (mode b)
//
// Per-pixel confidence maps are optional. Modes b and c pass them through to
// FusionIntegrator, which weights samples by `clamp(conf, 0, 1)` by default.
// Mode a integrates external sensor depth, so it never consumes a DA3
// confidence map and weighting is disabled automatically. An explicit binary
// mask remains available through `--conf-thresh` for callers that want flying
// pixels bounded before weighting.
//
// Integration is delegated to FusionIntegrator so this file stays independent
// of the Open3D/fallback backend split. The only artifact owned by this runner
// is `<out_dir>/fused_mesh.ply`; the callers keep writing poses.txt,
// intrinsics.txt and output_cloud.ply through their existing paths.

#include "depth_anything3_inference.hpp"
#include "fusion_integrator.hpp"
#include "fusion_utils.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <opencv2/core.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace da3::fusion {

/// Fusion experiment selected by `--fusion-mode a|b|c`.
enum class FusionMode : std::uint8_t { a, b, c };

/// Parse a `--fusion-mode` argument.
/// @throws std::runtime_error when the value is not one of `a`, `b`, `c`.
[[nodiscard]] inline FusionMode parse_fusion_mode(const std::string &name) {
    if (name == "a") {
        return FusionMode::a;
    }
    if (name == "b") {
        return FusionMode::b;
    }
    if (name == "c") {
        return FusionMode::c;
    }
    throw std::runtime_error("Unknown fusion mode: '" + name + "' (expected one of: a, b, c)");
}

/// Canonical lowercase CLI name for a FusionMode (inverse of parse_fusion_mode).
[[nodiscard]] inline std::string fusion_mode_name(FusionMode mode) {
    switch (mode) {
    case FusionMode::a:
        return "a";
    case FusionMode::b:
        return "b";
    case FusionMode::c:
        return "c";
    }
    return "a";
}

/// True when the mode takes its trajectory from an external `poses.txt`.
[[nodiscard]] inline bool uses_external_poses(FusionMode mode) noexcept { return mode != FusionMode::c; }

/// Outcome of one offline fusion run.
struct FusionRunSummary {
    std::size_t frames_attempted{0};
    std::size_t frames_integrated{0};
    std::size_t frames_skipped{0};
    std::size_t frames_scale_aligned{0};
    double mean_depth_scale{1.0};
    bool mesh_written{false};
    std::filesystem::path mesh_path;

    /// Contextual conversion for call sites that only check success.
    [[nodiscard]] explicit operator bool() const noexcept { return mesh_written; }
};

/// Convert a depth image to single-channel CV_32F metres.
///
/// `FrameData::depth` is already float metres; the extra branches keep the
/// helper usable for 16-bit millimetre capture depth supplied as a reference.
[[nodiscard]] inline bool to_metric_depth(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 1) {
        return false;
    }
    switch (src.depth()) {
    case CV_32F:
        dst = src;
        return true;
    case CV_64F:
        src.convertTo(dst, CV_32F);
        return true;
    case CV_16U:
        src.convertTo(dst, CV_32F, 0.001); // millimetres -> metres
        return true;
    case CV_8U:
        src.convertTo(dst, CV_32F);
        return true;
    default:
        return false;
    }
}

/// Read the external `poses.txt` used by modes a and b (12 w2c values per row).
/// @throws std::runtime_error on I/O or parse failure.
[[nodiscard]] inline std::vector<std::array<float, 12>> load_external_poses(const std::filesystem::path &poses_path) {
    std::vector<std::array<float, 12>> poses;
    if (!read_poses_txt(poses_path, poses)) {
        throw std::runtime_error("Could not read external poses from " + poses_path.string());
    }
    return poses;
}

namespace detail {

/// Default threshold for the opt-in binary confidence mask. Weighting is the
/// default path; this constant is retained so `--conf-thresh 0.5` reproduces
/// the historical mask cut when a caller explicitly asks for it.
inline constexpr float kConfidenceThreshold = 0.5F;
inline constexpr float kEdgeThresholdM = 0.05F; // Sobel gradient bound (metres)

/// Per-frame preprocessing result consumed by run_offline_fusion().
struct PreparedFrame {
    cv::Mat depth;
    cv::Mat valid;
    Pinhole intrinsics;
    double scale{1.0};
    bool scale_aligned{false};
};

/// Step 1: rescale calibration intrinsics to the resolution actually being
/// integrated. `FrameData::intrinsics` describes the DA3 depth resolution,
/// which differs from the capture depth in mode a. Returns false when the
/// rescaled focal length is unusable.
[[nodiscard]] inline bool scaled_calibration(const tasks::FrameData &frame, const cv::Mat &model_depth,
                                             bool have_model_depth, const cv::Mat &source_depth, Pinhole &out) {
    const int calibration_w = have_model_depth ? model_depth.cols : source_depth.cols;
    const int calibration_h = have_model_depth ? model_depth.rows : source_depth.rows;
    const Pinhole calibration{frame.intrinsics[0], frame.intrinsics[1], frame.intrinsics[2], frame.intrinsics[3]};
    out = scale_intrinsics(calibration, calibration_w, calibration_h, source_depth.cols, source_depth.rows);
    return out.fx > 0.0F && out.fy > 0.0F;
}

/// Step 4 (mode b only): median ratio between the metric reference and the DA3
/// depth. Returns 1.0 when there is nothing to align against, and rejects
/// non-positive or non-finite estimates.
[[nodiscard]] inline double align_depth_scale(const cv::Mat &depth, const cv::Mat &valid, const cv::Mat &reference) {
    if (reference.empty() || reference.size() != depth.size()) {
        return 1.0;
    }
    const double scale = estimate_scale_median(depth, reference, valid);
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return 1.0;
    }
    return scale;
}

/// Resolve the external reference for mode b and delegate the actual estimate.
[[nodiscard]] inline double maybe_align_depth_scale(FusionMode mode, const cv::Mat &depth, const cv::Mat &valid,
                                                    const std::vector<cv::Mat> &reference_depth, std::size_t index) {
    if (mode != FusionMode::b || index >= reference_depth.size()) {
        return 1.0;
    }
    cv::Mat reference;
    if (!to_metric_depth(reference_depth[index], reference)) {
        return 1.0;
    }
    return align_depth_scale(depth, valid, reference);
}

/// Multiply `scale` into the depth, reporting whether a conversion happened.
[[nodiscard]] inline bool apply_depth_scale(cv::Mat &depth, double scale) {
    if (std::abs(scale - 1.0) <= 1e-9) {
        return false;
    }
    cv::Mat scaled;
    depth.convertTo(scaled, CV_32F, scale);
    depth = scaled;
    return true;
}

/// Pick the depth actually integrated: external capture depth in mode a when it
/// converts, otherwise the DA3 depth. Returns nullptr when neither is usable.
[[nodiscard]] inline const cv::Mat *select_source_depth(FusionMode mode, std::size_t index,
                                                        const std::vector<cv::Mat> &reference_depth,
                                                        const cv::Mat &model_depth, bool have_model_depth,
                                                        cv::Mat &external_depth) {
    if (mode != FusionMode::a || index >= reference_depth.size()) {
        return have_model_depth ? &model_depth : nullptr;
    }
    if (!to_metric_depth(reference_depth[index], external_depth)) {
        return have_model_depth ? &model_depth : nullptr;
    }
    return &external_depth;
}

/// Run steps 1-4 for one frame in the documented order.
///
/// `confidence` is optional and must be at the source-depth resolution when
/// present. `conf_threshold` is the opt-in binary mask value; when it is unset
/// (the default) depth is left unchanged and confidence weighting happens
/// inside FusionIntegrator. The clone is intentional: later stages zero invalid
/// pixels and must not mutate the caller's `FrameData::depth`.
[[nodiscard]] inline bool prepare_frame(const tasks::FrameData &frame, const cv::Mat &model_depth,
                                        bool have_model_depth, const cv::Mat &source_depth, FusionMode mode,
                                        const std::vector<cv::Mat> &reference_depth, std::size_t index,
                                        PreparedFrame &out, const cv::Mat &confidence = cv::Mat(),
                                        std::optional<float> conf_threshold = std::nullopt) {
    if (!scaled_calibration(frame, model_depth, have_model_depth, source_depth, out.intrinsics)) {
        return false;
    }

    // Step 2: optional binary low-confidence mask. Weighting is the default
    // path, so an unset threshold leaves the depth untouched (the historical
    // empty-map behaviour) and the confidence map reaches the integrator.
    out.depth = source_depth.clone();
    if (conf_threshold.has_value() && !confidence.empty() && confidence.rows == source_depth.rows &&
        confidence.cols == source_depth.cols && confidence.channels() == 1) {
        cv::Mat conf32;
        if (confidence.type() == CV_32FC1) {
            conf32 = confidence;
        } else if (confidence.type() == CV_64FC1) {
            confidence.convertTo(conf32, CV_32F);
        }
        if (!conf32.empty()) {
            out.depth = mask_low_confidence(source_depth, conf32, *conf_threshold);
        }
    }

    // Step 3: reject non-positive and flying pixels.
    out.valid = compute_valid_mask(out.depth, kEdgeThresholdM);

    // Step 4: optional median scale alignment (mode b only). The external
    // capture depth is the metric reference; `estimate_scale_median` returns
    // ref/pred, so it is multiplied into the DA3 depth and the external
    // trajectory stays untouched.
    out.scale = maybe_align_depth_scale(mode, out.depth, out.valid, reference_depth, index);
    out.scale_aligned = apply_depth_scale(out.depth, out.scale);
    return true;
}

} // namespace detail

/// Fuse a sequence into `<out_dir>/fused_mesh.ply`.
///
/// @param frames          Postprocessed frames. `depth` comes from DA3 except in
///                        mode a, where an external depth pass is preferred.
/// @param poses_override  External w2c rows (36 values each) read with
///                        `read_poses_txt()`. Required by modes a and b; ignored
///                        by mode c, which uses `FrameData::pose`.
/// @param mode            Experiment selector.
/// @param out_dir         Destination directory for `fused_mesh.ply`.
/// @param reference_depth Optional per-frame metric external depth. Mode a uses
///                        it as the depth source; mode b uses it only as the
///                        `estimate_scale_median()` reference. Entries may be
///                        empty when a frame has no capture depth.
/// @param confidence      Optional per-frame confidence maps aligned with
///                        `frames` (typically `DepthResult::conf`). Only modes b
///                        and c consume them; mode a integrates sensor depth and
///                        disables weighting automatically. Empty entries fall
///                        back to uniform weight for that frame.
/// @param conf_threshold  Optional binary mask threshold. When set, depth below
///                        the threshold is zeroed before integration; when
///                        unset the runner relies on confidence weighting. This
///                        preserves the historical `--conf-thresh` escape hatch
///                        for bounding flying pixels.
///
/// @throws std::runtime_error on an empty sequence, a pose/frame count
///         mismatch, or a malformed external pose row.
[[nodiscard]] inline FusionRunSummary run_offline_fusion(const std::vector<tasks::FrameData> &frames,
                                                         const std::vector<std::array<float, 12>> &poses_override,
                                                         FusionMode mode, const std::filesystem::path &out_dir,
                                                         const std::vector<cv::Mat> &reference_depth = {},
                                                         const std::vector<cv::Mat> &confidence = {},
                                                         std::optional<float> conf_threshold = std::nullopt) {
    if (frames.empty()) {
        throw std::runtime_error("Fusion requires at least one frame");
    }
    if (uses_external_poses(mode) && poses_override.size() != frames.size()) {
        throw std::runtime_error("Fusion mode '" + fusion_mode_name(mode) + "' needs one external pose per frame (" +
                                 std::to_string(poses_override.size()) + " pose(s) for " +
                                 std::to_string(frames.size()) + " frame(s))");
    }

    FusionIntegrator::Options options;
    options.stride = 1;
    // Sensor depth in mode a has no matching DA3 confidence map; disable
    // weighting so mode a stays byte-compatible with the earlier fallback.
    options.conf_weighting = mode != FusionMode::a;
    FusionIntegrator integrator(options);

    FusionRunSummary summary;
    summary.mesh_path = out_dir / "fused_mesh.ply";
    double scale_sum = 0.0;
    std::size_t scale_samples = 0;
    const cv::Mat no_confidence;

    for (std::size_t i = 0; i < frames.size(); ++i) {
        ++summary.frames_attempted;
        const tasks::FrameData &frame = frames[i];

        cv::Mat model_depth;
        const bool have_model_depth = to_metric_depth(frame.depth, model_depth);

        // Mode a prefers the externally captured depth (D435 + ICP baseline).
        // Modes b and c always integrate the DA3 depth.
        cv::Mat external_depth;
        const cv::Mat *source =
            detail::select_source_depth(mode, i, reference_depth, model_depth, have_model_depth, external_depth);
        if (source == nullptr) {
            ++summary.frames_skipped;
            continue;
        }
        const cv::Mat &source_depth = *source;

        // Confidence is only meaningful for DA3 depth. Mode a deliberately
        // ignores it, and a stale/mismatched map is dropped rather than
        // refusing an otherwise valid frame.
        const cv::Mat *found_confidence = nullptr;
        if (mode != FusionMode::a && i < confidence.size() && !confidence[i].empty() &&
            confidence[i].rows == source_depth.rows && confidence[i].cols == source_depth.cols) {
            found_confidence = &confidence[i];
        }
        const cv::Mat &frame_confidence = found_confidence != nullptr ? *found_confidence : no_confidence;

        detail::PreparedFrame prepared;
        if (!detail::prepare_frame(frame, model_depth, have_model_depth, source_depth, mode, reference_depth, i,
                                   prepared, frame_confidence, conf_threshold)) {
            ++summary.frames_skipped;
            continue;
        }
        summary.frames_scale_aligned += prepared.scale_aligned ? 1 : 0;
        scale_sum += prepared.scale;
        ++scale_samples;

        // Keep the mask's "invalid" pixels out of the integrator so edge
        // artefacts do not become surface points. OpenCV's InputArray scalar
        // constructor takes a double, so widen the float fill value explicitly
        // rather than relying on an implicit float->double promotion.
        cv::Mat invalid;
        cv::bitwise_not(prepared.valid, invalid);
        prepared.depth.setTo(static_cast<double>(0.0F), invalid);

        const std::array<float, 12> pose = uses_external_poses(mode) ? poses_override[i] : frame.pose;

        if (!integrator.integrate(prepared.depth, frame.image_bgr, pose, prepared.intrinsics, frame_confidence)) {
            ++summary.frames_skipped;
            continue;
        }
        ++summary.frames_integrated;
    }

    summary.mean_depth_scale = scale_samples == 0 ? 1.0 : scale_sum / static_cast<double>(scale_samples);
    summary.mesh_written = integrator.extract_mesh(summary.mesh_path);
    return summary;
}

/// String overload for callers that keep the raw CLI value.
[[nodiscard]] inline FusionRunSummary run_offline_fusion(const std::vector<tasks::FrameData> &frames,
                                                         const std::vector<std::array<float, 12>> &poses_override,
                                                         const std::string &mode, const std::filesystem::path &out_dir,
                                                         const std::vector<cv::Mat> &reference_depth = {},
                                                         const std::vector<cv::Mat> &confidence = {},
                                                         std::optional<float> conf_threshold = std::nullopt) {
    return run_offline_fusion(frames, poses_override, parse_fusion_mode(mode), out_dir, reference_depth, confidence,
                              conf_threshold);
}

/// Character overload so `'a'`/`'b'`/`'c'` work at call sites.
[[nodiscard]] inline FusionRunSummary run_offline_fusion(const std::vector<tasks::FrameData> &frames,
                                                         const std::vector<std::array<float, 12>> &poses_override,
                                                         char mode, const std::filesystem::path &out_dir,
                                                         const std::vector<cv::Mat> &reference_depth = {},
                                                         const std::vector<cv::Mat> &confidence = {},
                                                         std::optional<float> conf_threshold = std::nullopt) {
    return run_offline_fusion(frames, poses_override, parse_fusion_mode(std::string(1, mode)), out_dir, reference_depth,
                              confidence, conf_threshold);
}

} // namespace da3::fusion
