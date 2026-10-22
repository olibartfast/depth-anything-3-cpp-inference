#pragma once

#include <cstdint>
#include <filesystem>
#include <opencv2/core.hpp>
#include <string>

/// Camera capture helpers. The capture loop is deliberately independent of any
/// SDK: hardware sources implement `Source`, and tests drive it with synthetic
/// frames. RealSense is one implementation behind `USE_REALSENSE`.
namespace da3::camera {

/// Pinhole color intrinsics in pixels, read from the active stream profile.
struct Intrinsics {
    float fx{0.0F};
    float fy{0.0F};
    float cx{0.0F};
    float cy{0.0F};
    int width{0};
    int height{0};

    [[nodiscard]] bool valid() const noexcept { return fx > 0.0F && fy > 0.0F && width > 0 && height > 0; }
};

/// Focal length used by the metric formula: the mean of fx and fy.
[[nodiscard]] float focal_px(const Intrinsics &intrinsics) noexcept;

/// One captured frame. `depth_mm` is empty when the source has no depth stream.
struct Frame {
    cv::Mat color_bgr; // CV_8UC3
    cv::Mat depth_mm;  // CV_16UC1, aligned to the color stream when present
    Intrinsics intrinsics;
    std::uint64_t timestamp_ms{0};
};

/// Blocking frame source. Implementations own their device session.
class Source {
  public:
    Source() = default;
    Source(const Source &) = delete;
    Source(Source &&) = delete;
    Source &operator=(const Source &) = delete;
    Source &operator=(Source &&) = delete;
    virtual ~Source() = default;

    [[nodiscard]] virtual std::string device_name() const = 0;

    /// Block until the next frame is available. Returns false when the source ends.
    virtual bool next(Frame &frame) = 0;
};

struct CaptureConfig {
    int frames{1};
    int width{1280};
    int height{720};
    int fps{30};
};

struct CaptureResult {
    int frames{0};
    int depth_frames{0};
    Intrinsics intrinsics;
    std::filesystem::path root; // contains color/, depth/, intrinsics.txt, capture.txt
};

/// Capture `config.frames` frames into `root`:
///   color/%03d.png   lossless BGR
///   depth/%03d.png   16-bit millimetres, aligned to color
///   intrinsics.txt   one `fx fy cx cy` row per frame (existing format)
///   capture.txt      device, size, counts, and per-frame timestamps
///
/// Throws std::runtime_error on an ended stream, invalid frames, or write failure.
/// The color and depth directories are separate so the image collector never
/// mistakes a depth map for an input view.
[[nodiscard]] CaptureResult capture_to_directory(Source &source, const CaptureConfig &config,
                                                 const std::filesystem::path &root);

} // namespace da3::camera
