#include "processing_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace da3::processing {

void normalize_image(std::span<float> data, size_t channel_size, std::span<const float, 3> means,
                     std::span<const float, 3> stds) {
    for (size_t c = 0; c < 3; ++c) {
        const float mean = means[c];
        const float std_val = stds[c];
        for (size_t i = 0; i < channel_size; ++i) {
            data[c * channel_size + i] = (data[c * channel_size + i] - mean) / std_val;
        }
    }
}

int round_to_multiple(int value, int multiple) noexcept {
    if (multiple <= 0) {
        return value;
    }
    const int lower = (value / multiple) * multiple;
    const int upper = lower + multiple;
    return (std::abs(upper - value) <= std::abs(value - lower)) ? upper : lower;
}

std::pair<int, int> compute_output_size(int in_h, int in_w, int process_res, int patch_size) noexcept {
    // Resize so the longest side equals process_res (preserve aspect ratio)
    const float scale = static_cast<float>(process_res) / static_cast<float>(std::max(in_h, in_w));
    const int scaled_h = std::max(1, static_cast<int>(std::round(static_cast<float>(in_h) * scale)));
    const int scaled_w = std::max(1, static_cast<int>(std::round(static_cast<float>(in_w) * scale)));

    // Round each dimension to the nearest multiple of patch_size
    const int out_h = std::max(patch_size, round_to_multiple(scaled_h, patch_size));
    const int out_w = std::max(patch_size, round_to_multiple(scaled_w, patch_size));

    return {out_h, out_w};
}

void preprocess_frame(const cv::Mat &bgr_frame, std::span<float> output, int out_h, int out_w,
                      std::span<const float, 3> means, std::span<const float, 3> stds) {
    const auto h = static_cast<size_t>(out_h);
    const auto w = static_cast<size_t>(out_w);

    if (output.size() < 3 * h * w) {
        throw std::invalid_argument("Output buffer too small for preprocess_frame");
    }

    cv::Mat resized;
    cv::resize(bgr_frame, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(resized, channels);

    float *ptr = output.data();
    for (size_t c = 0; c < 3; ++c) {
        std::memcpy(ptr, channels[c].data, h * w * sizeof(float));
        ptr += h * w;
    }

    normalize_image(output.subspan(0, 3 * h * w), h * w, means, stds);
}

cv::Mat apply_depth_colormap(const cv::Mat &depth_map) {
    // Normalize to [0, 255]
    cv::Mat normalized;
    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(depth_map, &min_val, &max_val);

    if (max_val - min_val < 1e-6) {
        // Uniform depth — return a grey image
        normalized = cv::Mat::zeros(depth_map.size(), CV_8U);
        normalized.setTo(128);
    } else {
        depth_map.convertTo(normalized, CV_8U, 255.0 / (max_val - min_val), -255.0 * min_val / (max_val - min_val));
    }

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_INFERNO);
    return colored;
}

cv::Mat apply_conf_colormap(const cv::Mat &conf_map) {
    cv::Mat normalized;
    conf_map.convertTo(normalized, CV_8U, 255.0);
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);
    return colored;
}

float metric_depth_from_relative(float net_output, float focal_px) noexcept {
    // metric (meters) = focal_px * relative / 300
    // NOTE: Nested model output is already in meters — do not scale it.
    return focal_px * net_output / 300.f;
}

} // namespace da3::processing
