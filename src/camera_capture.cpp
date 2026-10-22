#include "camera_capture.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace da3::camera {
namespace {

std::string frame_name(int index, const char *extension) {
    std::ostringstream name;
    name << std::setw(3) << std::setfill('0') << index << extension;
    return name.str();
}

cv::Mat to_bgr(const cv::Mat &image) {
    if (image.type() == CV_8UC3) {
        return image;
    }
    cv::Mat bgr;
    if (image.type() == CV_8UC4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else if (image.type() == CV_8UC1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        throw std::runtime_error("Capture source returned an unsupported color format (type " +
                                 std::to_string(image.type()) + ")");
    }
    return bgr;
}

void write_metadata(const std::filesystem::path &path, const std::string &device, const CaptureConfig &config,
                    const CaptureResult &result, const std::vector<std::uint64_t> &timestamps) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not write capture metadata to " + path.string());
    }
    out << "device: " << device << '\n';
    out << "requested_size: " << config.width << 'x' << config.height << '\n';
    out << "requested_fps: " << config.fps << '\n';
    out << "frames: " << result.frames << '\n';
    out << "depth_frames: " << result.depth_frames << '\n';
    out << "color_size: " << result.intrinsics.width << 'x' << result.intrinsics.height << '\n';
    out << "depth_unit: millimetre\n";
    out << "depth_aligned_to: color\n";
    out << std::setprecision(6);
    for (std::size_t i = 0; i < timestamps.size(); ++i) {
        out << "frame " << frame_name(static_cast<int>(i), "") << " timestamp_ms " << timestamps[i] << '\n';
    }
    if (!out) {
        throw std::runtime_error("Could not finish writing capture metadata to " + path.string());
    }
}

} // namespace

float focal_px(const Intrinsics &intrinsics) noexcept { return 0.5F * (intrinsics.fx + intrinsics.fy); }

CaptureResult capture_to_directory(Source &source, const CaptureConfig &config, const std::filesystem::path &root) {
    if (config.frames <= 0) {
        throw std::invalid_argument("Capture frame count must be positive");
    }
    if (config.width <= 0 || config.height <= 0) {
        throw std::invalid_argument("Capture size must be positive");
    }
    if (config.fps <= 0) {
        throw std::invalid_argument("Capture fps must be positive");
    }

    const std::filesystem::path color_dir = root / "color";
    const std::filesystem::path depth_dir = root / "depth";
    std::filesystem::create_directories(color_dir);
    std::filesystem::create_directories(depth_dir);

    CaptureResult result;
    result.root = root;
    std::vector<Intrinsics> intrinsics;
    std::vector<std::uint64_t> timestamps;
    intrinsics.reserve(static_cast<std::size_t>(config.frames));
    timestamps.reserve(static_cast<std::size_t>(config.frames));

    for (int index = 0; index < config.frames; ++index) {
        Frame frame;
        if (!source.next(frame)) {
            throw std::runtime_error("Capture stream ended after " + std::to_string(index) + " of " +
                                     std::to_string(config.frames) + " frame(s)");
        }
        if (frame.color_bgr.empty()) {
            throw std::runtime_error("Capture source returned an empty color frame");
        }
        if (!frame.intrinsics.valid()) {
            throw std::runtime_error("Capture source returned invalid intrinsics");
        }

        const cv::Mat color = to_bgr(frame.color_bgr);
        if (index == 0) {
            result.intrinsics = frame.intrinsics;
        }
        const std::string name = frame_name(index, ".png");
        if (!cv::imwrite((color_dir / name).string(), color)) {
            throw std::runtime_error("Could not write color frame to " + (color_dir / name).string());
        }

        if (!frame.depth_mm.empty()) {
            if (frame.depth_mm.type() != CV_16UC1) {
                throw std::runtime_error("Capture depth frame must be 16-bit single-channel (type " +
                                         std::to_string(frame.depth_mm.type()) + ")");
            }
            if (frame.depth_mm.size() != color.size()) {
                throw std::runtime_error("Capture depth frame is not aligned with the color frame");
            }
            if (!cv::imwrite((depth_dir / name).string(), frame.depth_mm)) {
                throw std::runtime_error("Could not write depth frame to " + (depth_dir / name).string());
            }
            ++result.depth_frames;
        }

        intrinsics.push_back(frame.intrinsics);
        timestamps.push_back(frame.timestamp_ms);
        ++result.frames;
    }

    const std::filesystem::path intrinsics_path = root / "intrinsics.txt";
    std::ofstream intrinsics_out(intrinsics_path);
    if (!intrinsics_out) {
        throw std::runtime_error("Could not write intrinsics to " + intrinsics_path.string());
    }
    intrinsics_out << std::setprecision(6);
    for (const Intrinsics &intrinsic : intrinsics) {
        intrinsics_out << intrinsic.fx << ' ' << intrinsic.fy << ' ' << intrinsic.cx << ' ' << intrinsic.cy << '\n';
    }
    if (!intrinsics_out) {
        throw std::runtime_error("Could not finish writing intrinsics to " + intrinsics_path.string());
    }

    write_metadata(root / "capture.txt", source.device_name(), config, result, timestamps);
    return result;
}

} // namespace da3::camera
