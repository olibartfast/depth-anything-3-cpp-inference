#include "backends/inference_backend.hpp"
#include "depth_anything3_inference.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#ifndef DA3_TEST_FIXTURE_DIR
#error "DA3_TEST_FIXTURE_DIR must be defined by CMake"
#endif

#ifdef USE_ONNX_RUNTIME

namespace {

std::filesystem::path tiny_model() { return std::filesystem::path(DA3_TEST_FIXTURE_DIR) / "tiny_depth.onnx"; }

} // namespace

TEST(OnnxBackend, MissingModelThrows) {
    auto backend = da3::backend::create_backend();
    EXPECT_THROW((void)backend->initialize(tiny_model().parent_path() / "missing.onnx", {1, 3, 0, 0}),
                 std::runtime_error);
}

TEST(OnnxBackend, ConstantChannelMean) {
    auto backend = da3::backend::create_backend();
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->get_backend_name(), "ONNX Runtime");

    const auto detected = backend->initialize(tiny_model(), {1, 3, 0, 0});
    // Dynamic-axes models keep the caller's {1,3,0,0} placeholder.
    EXPECT_EQ(detected, (std::vector<int64_t>{1, 3, 0, 0}));
    ASSERT_EQ(backend->get_output_count(), 1u);

    constexpr int h = 4;
    constexpr int w = 4;
    std::vector<float> input(static_cast<size_t>(3 * h * w));
    for (int c = 0; c < 3; ++c) {
        std::fill_n(input.begin() + static_cast<std::ptrdiff_t>(c * h * w), h * w, static_cast<float>(c + 1));
    }

    backend->run_inference(input, {1, 3, h, w});
    EXPECT_EQ(backend->get_output_shape(0), (std::vector<int64_t>{1, h, w}));

    std::vector<float> output(static_cast<size_t>(h * w));
    backend->get_output_data(0, output.data(), output.size());
    for (float value : output) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    EXPECT_THROW((void)backend->get_output_shape(1), std::out_of_range);
    EXPECT_THROW(backend->get_output_data(0, output.data(), output.size() - 1), std::runtime_error);
}

TEST(OnnxBackend, HighLevelInferenceUsesModel) {
    Config config;
    config.process_res = 28;
    config.patch_size = 14;

    DepthAnything3Inference inference(tiny_model(), config);

    cv::Mat image(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
    int orig_h = 0;
    int orig_w = 0;
    int out_h = 0;
    int out_w = 0;
    const auto tensor = inference.preprocess_image(image, orig_h, orig_w, out_h, out_w);

    EXPECT_EQ(orig_h, 8);
    EXPECT_EQ(orig_w, 8);
    EXPECT_EQ(out_h, 28);
    EXPECT_EQ(out_w, 28);
    EXPECT_EQ(tensor.size(), static_cast<size_t>(3 * 28 * 28));

    inference.run_inference(tensor, out_h, out_w);
    const auto result = inference.postprocess_outputs(out_h, out_w, orig_h, orig_w, true);
    ASSERT_EQ(result.depth.rows, orig_h);
    ASSERT_EQ(result.depth.cols, orig_w);

    // The fixture returns the mean of the normalized RGB channels.
    const auto normalized = [](float channel, float mean, float stddev) { return (channel / 255.0f - mean) / stddev; };
    const float expected =
        (normalized(30.0f, config.means[0], config.stds[0]) + normalized(20.0f, config.means[1], config.stds[1]) +
         normalized(10.0f, config.means[2], config.stds[2])) /
        3.0f;
    for (int row = 0; row < result.depth.rows; ++row) {
        for (int col = 0; col < result.depth.cols; ++col) {
            EXPECT_NEAR(result.depth.at<float>(row, col), expected, 1e-5f);
        }
    }
}

#else

TEST(OnnxBackend, SkippedWithoutOnnxRuntime) {
    GTEST_SKIP() << "Build with -DUSE_ONNX_RUNTIME=ON to run ONNX backend tests";
}

#endif
