#include "camera_capture.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>

namespace {

/// Synthetic camera source: no hardware, deterministic frames.
class FakeSource final : public da3::camera::Source {
  public:
    FakeSource(int frames, bool with_depth) : frames_(frames), with_depth_(with_depth) {}

    [[nodiscard]] std::string device_name() const override { return "Fake Camera"; }

    bool next(da3::camera::Frame &frame) override {
        if (index_ >= frames_) {
            return false;
        }
        frame.color_bgr = cv::Mat(4, 6, CV_8UC3, cv::Scalar(10.0 + index_, 20.0, 30.0));
        if (with_depth_) {
            frame.depth_mm = cv::Mat(4, 6, CV_16UC1, cv::Scalar(1000 + index_));
        }
        frame.intrinsics = {600.0F, 620.0F, 320.0F, 240.0F, 6, 4};
        frame.timestamp_ms = 1000U + static_cast<std::uint64_t>(index_);
        ++index_;
        return true;
    }

  private:
    int frames_;
    bool with_depth_;
    int index_{0};
};

/// Temporary capture directory, removed on scope exit.
class TempDir {
  public:
    TempDir() {
        root_ = std::filesystem::temp_directory_path() /
                ("da3_capture_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    }

    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;

    [[nodiscard]] const std::filesystem::path &path() const { return root_; }

  private:
    std::filesystem::path root_;
};

std::size_t count_files(const std::filesystem::path &directory) {
    std::size_t count = 0;
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            ++count;
        }
    }
    return count;
}

std::string read_text(const std::filesystem::path &path) {
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

} // namespace

TEST(CameraIntrinsics, FocalIsMeanOfFxAndFy) {
    const da3::camera::Intrinsics intrinsics{600.0F, 620.0F, 320.0F, 240.0F, 640, 480};
    EXPECT_FLOAT_EQ(da3::camera::focal_px(intrinsics), 610.0F);
}

TEST(CameraIntrinsics, ValidityRequiresPositiveFocalAndSize) {
    EXPECT_TRUE((da3::camera::Intrinsics{600.0F, 600.0F, 320.0F, 240.0F, 640, 480}.valid()));
    EXPECT_FALSE((da3::camera::Intrinsics{0.0F, 600.0F, 320.0F, 240.0F, 640, 480}.valid()));
    EXPECT_FALSE((da3::camera::Intrinsics{600.0F, 600.0F, 320.0F, 240.0F, 0, 480}.valid()));
}

TEST(CameraCapture, WritesColorDepthIntrinsicsAndMetadata) {
    TempDir dir;
    FakeSource source(2, true);
    const da3::camera::CaptureConfig config{2, 640, 480, 30};

    const da3::camera::CaptureResult result = da3::camera::capture_to_directory(source, config, dir.path());

    EXPECT_EQ(result.frames, 2);
    EXPECT_EQ(result.depth_frames, 2);
    EXPECT_FLOAT_EQ(result.intrinsics.fx, 600.0F);
    EXPECT_EQ(result.intrinsics.width, 6);
    EXPECT_EQ(result.intrinsics.height, 4);

    EXPECT_EQ(count_files(dir.path() / "color"), 2U);
    EXPECT_EQ(count_files(dir.path() / "depth"), 2U);

    const cv::Mat color = cv::imread((dir.path() / "color" / "000.png").string(), cv::IMREAD_COLOR);
    ASSERT_FALSE(color.empty());
    EXPECT_EQ(color.type(), CV_8UC3);
    EXPECT_EQ(color.size(), cv::Size(6, 4));

    const cv::Mat depth = cv::imread((dir.path() / "depth" / "001.png").string(), cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(depth.empty());
    EXPECT_EQ(depth.type(), CV_16UC1);
    EXPECT_EQ(depth.at<std::uint16_t>(0, 0), 1001U);

    const std::string intrinsics = read_text(dir.path() / "intrinsics.txt");
    EXPECT_NE(intrinsics.find("600 620 320 240"), std::string::npos);

    const std::string metadata = read_text(dir.path() / "capture.txt");
    EXPECT_NE(metadata.find("device: Fake Camera"), std::string::npos);
    EXPECT_NE(metadata.find("frames: 2"), std::string::npos);
    EXPECT_NE(metadata.find("depth_unit: millimetre"), std::string::npos);
}

TEST(CameraCapture, ColorOnlyLeavesDepthDirectoryEmpty) {
    TempDir dir;
    FakeSource source(1, false);
    const da3::camera::CaptureConfig config{1, 640, 480, 30};

    const da3::camera::CaptureResult result = da3::camera::capture_to_directory(source, config, dir.path());

    EXPECT_EQ(result.frames, 1);
    EXPECT_EQ(result.depth_frames, 0);
    EXPECT_EQ(count_files(dir.path() / "depth"), 0U);
    EXPECT_EQ(count_files(dir.path() / "color"), 1U);
}

TEST(CameraCapture, RejectsNonPositiveFrameCount) {
    TempDir dir;
    FakeSource source(1, true);
    EXPECT_THROW((void)da3::camera::capture_to_directory(source, {0, 640, 480, 30}, dir.path()), std::invalid_argument);
}

TEST(CameraCapture, RejectsNonPositiveFps) {
    TempDir dir;
    FakeSource source(1, true);
    EXPECT_THROW((void)da3::camera::capture_to_directory(source, {1, 640, 480, 0}, dir.path()), std::invalid_argument);
}

TEST(CameraCapture, ThrowsWhenStreamEndsEarly) {
    TempDir dir;
    FakeSource source(1, true);
    EXPECT_THROW((void)da3::camera::capture_to_directory(source, {2, 640, 480, 30}, dir.path()), std::runtime_error);
}

TEST(CameraCapture, RejectsMismatchedDepth) {
    class MismatchedDepthSource final : public da3::camera::Source {
      public:
        [[nodiscard]] std::string device_name() const override { return "Fake"; }
        bool next(da3::camera::Frame &frame) override {
            frame.color_bgr = cv::Mat(4, 6, CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
            frame.depth_mm = cv::Mat(2, 2, CV_16UC1, cv::Scalar(0));
            frame.intrinsics = {600.0F, 600.0F, 3.0F, 2.0F, 6, 4};
            return true;
        }
    } source;

    TempDir dir;
    EXPECT_THROW((void)da3::camera::capture_to_directory(source, {1, 640, 480, 30}, dir.path()), std::runtime_error);
}

TEST(CameraCapture, ConvertsGrayscaleAndBgraColor) {
    class GraySource final : public da3::camera::Source {
      public:
        [[nodiscard]] std::string device_name() const override { return "Gray"; }
        bool next(da3::camera::Frame &frame) override {
            frame.color_bgr = cv::Mat(4, 6, CV_8UC1, cv::Scalar(128));
            frame.intrinsics = {600.0F, 600.0F, 3.0F, 2.0F, 6, 4};
            return true;
        }
    } gray;

    class BgraSource final : public da3::camera::Source {
      public:
        [[nodiscard]] std::string device_name() const override { return "BGRA"; }
        bool next(da3::camera::Frame &frame) override {
            frame.color_bgr = cv::Mat(4, 6, CV_8UC4, cv::Scalar(1.0, 2.0, 3.0, 4.0));
            frame.intrinsics = {600.0F, 600.0F, 3.0F, 2.0F, 6, 4};
            return true;
        }
    } bgra;

    TempDir gray_dir;
    EXPECT_EQ(da3::camera::capture_to_directory(gray, {1, 640, 480, 30}, gray_dir.path()).frames, 1);
    TempDir bgra_dir;
    EXPECT_EQ(da3::camera::capture_to_directory(bgra, {1, 640, 480, 30}, bgra_dir.path()).frames, 1);

    EXPECT_EQ(cv::imread((gray_dir.path() / "color" / "000.png").string(), cv::IMREAD_COLOR).type(), CV_8UC3);
    EXPECT_EQ(cv::imread((bgra_dir.path() / "color" / "000.png").string(), cv::IMREAD_COLOR).type(), CV_8UC3);
}
