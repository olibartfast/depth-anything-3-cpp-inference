#include "realsense_source.hpp"

#include <cstddef>
#include <cstdint>
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

namespace da3::camera {
namespace {

constexpr const char *kUdevHint =
    " If the camera is connected but inaccessible, install the librealsense udev rules "
    "(config/99-realsense-libusb.rules), reload rules with 'sudo udevadm control --reload-rules "
    "&& sudo udevadm trigger', then replug the camera.";

class RealsenseSource final : public Source {
  public:
    explicit RealsenseSource(const CaptureConfig &config) : align_to_color_(RS2_STREAM_COLOR) {
        rs2::config rs_config;
        rs_config.enable_stream(RS2_STREAM_COLOR, 0, config.width, config.height, RS2_FORMAT_BGR8, config.fps);
        // width/height 0 let the SDK pick a profile the device supports at this rate;
        // alignment resamples it to the color resolution.
        rs_config.enable_stream(RS2_STREAM_DEPTH, 0, 0, 0, RS2_FORMAT_Z16, config.fps);

        try {
            profile_ = pipeline_.start(rs_config);
        } catch (const rs2::error &error) {
            std::string message = std::string("Failed to start the RealSense pipeline: ") + error.what();
            const std::string detail = error.what();
            if (detail.find("permission") != std::string::npos || detail.find("LIBUSB") != std::string::npos ||
                detail.find("No device") != std::string::npos) {
                message += ".";
                message += kUdevHint;
            }
            throw std::runtime_error(message);
        }

        const auto color_stream = profile_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        const rs2_intrinsics intrinsics = color_stream.get_intrinsics();
        intrinsics_ = {intrinsics.fx,  intrinsics.fy,    intrinsics.ppx,
                       intrinsics.ppy, intrinsics.width, intrinsics.height};

        // The color sensor's auto-exposure needs a few frames to converge;
        // capturing immediately yields nearly black frames. Discard a short
        // warm-up so the requested frames are exposed like the live preview.
        constexpr int kWarmupFrames = 30;
        for (int index = 0; index < kWarmupFrames; ++index) {
            try {
                pipeline_.wait_for_frames();
            } catch (const rs2::error &) {
                break;
            }
        }

        try {
            device_name_ = profile_.get_device().get_info(RS2_CAMERA_INFO_NAME);
        } catch (const rs2::error &) {
            device_name_ = "Intel RealSense";
        }
    }

    [[nodiscard]] std::string device_name() const override { return device_name_; }

    bool next(Frame &frame) override {
        rs2::frameset frames;
        try {
            frames = pipeline_.wait_for_frames();
        } catch (const rs2::error &error) {
            throw std::runtime_error(std::string("RealSense frame capture failed: ") + error.what());
        }

        rs2::frameset aligned;
        try {
            aligned = align_to_color_.process(frames);
        } catch (const rs2::error &error) {
            throw std::runtime_error(std::string("RealSense depth/color alignment failed: ") + error.what());
        }
        const rs2::video_frame color = aligned.get_color_frame();
        const rs2::depth_frame depth = aligned.get_depth_frame();
        if (!color || !depth) {
            return false;
        }

        const int width = color.get_width();
        const int height = color.get_height();
        const cv::Mat color_view(height, width, CV_8UC3, const_cast<void *>(color.get_data()),
                                 static_cast<std::size_t>(color.get_stride_in_bytes()));
        frame.color_bgr = color_view.clone();

        const int depth_width = depth.get_width();
        const int depth_height = depth.get_height();
        if (depth_width != width || depth_height != height) {
            throw std::runtime_error("RealSense alignment returned a depth frame that does not match the color frame");
        }
        const cv::Mat depth_view(depth_height, depth_width, CV_16UC1, const_cast<void *>(depth.get_data()),
                                 static_cast<std::size_t>(depth.get_stride_in_bytes()));
        const float millimetres_per_unit = depth.get_units() * 1000.0F;
        frame.depth_mm.create(depth_height, depth_width, CV_16UC1);
        for (int y = 0; y < depth_height; ++y) {
            const auto *source = depth_view.ptr<std::uint16_t>(y);
            auto *target = frame.depth_mm.ptr<std::uint16_t>(y);
            for (int x = 0; x < depth_width; ++x) {
                target[x] = cv::saturate_cast<std::uint16_t>(static_cast<float>(source[x]) * millimetres_per_unit);
            }
        }

        frame.intrinsics = intrinsics_;
        frame.timestamp_ms = static_cast<std::uint64_t>(frames.get_timestamp());
        return true;
    }

  private:
    rs2::pipeline pipeline_;
    rs2::pipeline_profile profile_;
    rs2::align align_to_color_;
    Intrinsics intrinsics_;
    std::string device_name_;
};

} // namespace

std::unique_ptr<Source> make_realsense_source(const CaptureConfig &config) {
    return std::make_unique<RealsenseSource>(config);
}

} // namespace da3::camera
