#pragma once

#include <array>
#include <cstddef>
#include <opencv2/core.hpp>
#include <span>

namespace da3::processing {

/// Normalize CHW image data in-place: (pixel - mean) / std per channel
void normalize_image(std::span<float> data, size_t channel_size, std::span<const float, 3> means,
                     std::span<const float, 3> stds);

/// Round up to the nearest multiple of 'multiple'
[[nodiscard]] int round_to_multiple(int value, int multiple) noexcept;

/// Preprocess a BGR cv::Mat frame into a pre-allocated CHW float tensor.
/// output must have size >= 3 * out_h * out_w.
/// Performs: resize → BGR2RGB → float32 [0,1] → CHW split → normalize.
/// The output dimensions (out_h, out_w) are derived by resizing the longest
/// side to process_res and rounding to the nearest multiple of patch_size.
void preprocess_frame(const cv::Mat &bgr_frame, std::span<float> output, int out_h, int out_w,
                      std::span<const float, 3> means, std::span<const float, 3> stds);

/// Compute the output height and width given an input image, a target
/// resolution (longest-side bound), and the patch divisibility requirement.
/// @param in_h   Input image height
/// @param in_w   Input image width
/// @param process_res  Target size for the longest side
/// @param patch_size   Patch divisibility (default: 14 for DINOv2 backbone)
/// @returns {out_h, out_w}
[[nodiscard]] std::pair<int, int> compute_output_size(int in_h, int in_w, int process_res,
                                                      int patch_size = 14) noexcept;

/// Apply a colormap to a single-channel depth image for visualization.
/// Normalizes depth values, applies COLORMAP_INFERNO, and returns a BGR image.
[[nodiscard]] cv::Mat apply_depth_colormap(const cv::Mat &depth_map);

/// Apply a colormap to a single-channel confidence image for visualization.
/// Values are expected in [0, 1]; applies COLORMAP_JET.
[[nodiscard]] cv::Mat apply_conf_colormap(const cv::Mat &conf_map);

/// Convert a relative depth map to metric depth.
/// metric = focal_px * net_output / 300.f
/// NOTE: Nested model output is already in meters — do not scale it.
/// @param net_output Relative depth value from the network
/// @param focal_px   Focal length in pixels
[[nodiscard]] float metric_depth_from_relative(float net_output, float focal_px) noexcept;

} // namespace da3::processing
