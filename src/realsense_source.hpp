#pragma once

#include "camera_capture.hpp"

#include <memory>

namespace da3::camera {

/// Open the first Intel RealSense device and stream color plus depth.
/// Throws std::runtime_error when no device is available or the pipeline cannot
/// start. The message includes the udev-rules remedy for permission failures.
[[nodiscard]] std::unique_ptr<Source> make_realsense_source(const CaptureConfig &config);

} // namespace da3::camera
