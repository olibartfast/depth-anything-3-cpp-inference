#include "mock_backend.hpp"
#include "depth_anything3_inference.hpp"
#include "processing_utils.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

// ============================================================================
// normalize_image tests
// ============================================================================

TEST(NormalizeImage, AppliesMeanStd) {
    // 1 pixel per channel, 3 channels
    std::vector<float> data = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> means = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds = {0.229f, 0.224f, 0.225f};

    da3::processing::normalize_image(data, 1, means, stds);

    EXPECT_NEAR(data[0], (0.5f - 0.485f) / 0.229f, 1e-5f);
    EXPECT_NEAR(data[1], (0.5f - 0.456f) / 0.224f, 1e-5f);
    EXPECT_NEAR(data[2], (0.5f - 0.406f) / 0.225f, 1e-5f);
}

TEST(NormalizeImage, AllChannels) {
    // 2 pixels per channel, 3 channels = 6 floats
    std::vector<float> data = {
        0.1f, 0.2f,  // channel 0
        0.3f, 0.4f,  // channel 1
        0.5f, 0.6f   // channel 2
    };
    std::array<float, 3> means = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> stds = {0.5f, 0.25f, 0.1f};

    da3::processing::normalize_image(data, 2, means, stds);

    // Channel 0: divide by 0.5
    EXPECT_NEAR(data[0], 0.2f, 1e-5f);
    EXPECT_NEAR(data[1], 0.4f, 1e-5f);
    // Channel 1: divide by 0.25
    EXPECT_NEAR(data[2], 1.2f, 1e-5f);
    EXPECT_NEAR(data[3], 1.6f, 1e-5f);
    // Channel 2: divide by 0.1
    EXPECT_NEAR(data[4], 5.0f, 1e-5f);
    EXPECT_NEAR(data[5], 6.0f, 1e-5f);
}

// ============================================================================
// compute_output_size tests
// ============================================================================

TEST(ComputeOutputSize, LandscapeImage) {
    // 640x480, process_res=504, patch_size=14
    auto [h, w] = da3::processing::compute_output_size(480, 640, 504, 14);
    // Longest side is width (640) → scale = 504/640 ≈ 0.7875
    // Scaled: h ≈ 378, w = 504
    // Round to nearest multiple of 14: h=378 (378%14==0), w=504 (504%14==0)
    EXPECT_EQ(h % 14, 0);
    EXPECT_EQ(w % 14, 0);
    EXPECT_GE(h, 14);
    EXPECT_GE(w, 14);
    // Longer side should equal process_res after round-to-multiple
    EXPECT_EQ(w, 504);
}

TEST(ComputeOutputSize, PortraitImage) {
    // 480x640, process_res=504, patch_size=14
    auto [h, w] = da3::processing::compute_output_size(640, 480, 504, 14);
    EXPECT_EQ(h % 14, 0);
    EXPECT_EQ(w % 14, 0);
    EXPECT_EQ(h, 504);
}

TEST(ComputeOutputSize, SquareImage) {
    auto [h, w] = da3::processing::compute_output_size(512, 512, 504, 14);
    EXPECT_EQ(h % 14, 0);
    EXPECT_EQ(w % 14, 0);
    EXPECT_EQ(h, w);
}

TEST(ComputeOutputSize, DivisibilityGuaranteed) {
    // Run many combinations and check divisibility
    for (int in_h : {100, 300, 480, 720, 1080}) {
        for (int in_w : {100, 200, 640, 960, 1920}) {
            auto [oh, ow] = da3::processing::compute_output_size(in_h, in_w, 504, 14);
            EXPECT_EQ(oh % 14, 0) << "in=(" << in_h << "," << in_w << ")";
            EXPECT_EQ(ow % 14, 0) << "in=(" << in_h << "," << in_w << ")";
        }
    }
}

// ============================================================================
// round_to_multiple tests
// ============================================================================

TEST(RoundToMultiple, Exact) { EXPECT_EQ(da3::processing::round_to_multiple(14, 14), 14); }

TEST(RoundToMultiple, RoundsDown) { EXPECT_EQ(da3::processing::round_to_multiple(15, 14), 14); }

TEST(RoundToMultiple, RoundsUp) { EXPECT_EQ(da3::processing::round_to_multiple(22, 14), 28); }

TEST(RoundToMultiple, Zero) { EXPECT_EQ(da3::processing::round_to_multiple(0, 14), 0); }

// ============================================================================
// PreprocessFrame tests
// ============================================================================

TEST(PreprocessFrame, OutputDimensions) {
    cv::Mat img(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    const int out_h = 98;
    const int out_w = 196;
    std::vector<float> tensor(static_cast<size_t>(3 * out_h * out_w));
    std::array<float, 3> means = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds = {0.229f, 0.224f, 0.225f};

    da3::processing::preprocess_frame(img, tensor, out_h, out_w, means, stds);

    for (float v : tensor) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(PreprocessFrame, BufferTooSmall) {
    cv::Mat img(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<float> small_buf(10); // much too small
    std::array<float, 3> means = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> stds = {1.0f, 1.0f, 1.0f};

    EXPECT_THROW(da3::processing::preprocess_frame(img, small_buf, 100, 200, means, stds), std::invalid_argument);
}

// ============================================================================
// apply_depth_colormap tests
// ============================================================================

TEST(ApplyDepthColormap, OutputIsRGB) {
    cv::Mat depth(100, 100, CV_32F);
    cv::randu(depth, cv::Scalar(0.5f), cv::Scalar(10.0f));
    cv::Mat colored = da3::processing::apply_depth_colormap(depth);
    EXPECT_EQ(colored.rows, 100);
    EXPECT_EQ(colored.cols, 100);
    EXPECT_EQ(colored.channels(), 3);
}

TEST(ApplyDepthColormap, UniformDepth) {
    // All-same depth should not crash
    cv::Mat depth(50, 50, CV_32F, cv::Scalar(5.0f));
    EXPECT_NO_THROW({ auto _ = da3::processing::apply_depth_colormap(depth); });
}

// ============================================================================
// DepthAnything3Inference — constructor & config tests
// ============================================================================

class InferenceTest : public ::testing::Test {
  protected:
    // Build an inference instance using a MockBackend with a single depth output
    std::unique_ptr<DepthAnything3Inference> make_inference(int out_h = 36, int out_w = 36,
                                                            bool with_conf = false) {
        Config cfg;
        cfg.process_res = 504;
        cfg.output_conf = with_conf;

        const int npix = out_h * out_w;
        std::vector<float> depth_data(static_cast<size_t>(npix), 1.0f);

        std::vector<std::vector<float>> outputs = {std::move(depth_data)};
        std::vector<std::vector<int64_t>> shapes = {{1, static_cast<int64_t>(out_h), static_cast<int64_t>(out_w)}};

        if (with_conf) {
            std::vector<float> conf_data(static_cast<size_t>(npix), 0.9f);
            outputs.push_back(std::move(conf_data));
            shapes.push_back({1, static_cast<int64_t>(out_h), static_cast<int64_t>(out_w)});
        }

        auto backend = std::make_unique<MockBackend>();
        backend->set_outputs(std::move(outputs), std::move(shapes));
        return std::make_unique<DepthAnything3Inference>(std::move(backend), cfg);
    }
};

TEST_F(InferenceTest, ConfigAccessors) {
    auto inf = make_inference();
    EXPECT_EQ(inf->get_process_res(), 504);
    EXPECT_EQ(inf->get_patch_size(), 14);
}

TEST_F(InferenceTest, PreprocessFromMat) {
    auto inf = make_inference();

    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    int orig_h = 0;
    int orig_w = 0;
    int out_h = 0;
    int out_w = 0;

    auto tensor = inf->preprocess_image(img, orig_h, orig_w, out_h, out_w);

    EXPECT_EQ(orig_h, 480);
    EXPECT_EQ(orig_w, 640);
    EXPECT_EQ(out_h % 14, 0);
    EXPECT_EQ(out_w % 14, 0);
    EXPECT_EQ(tensor.size(), static_cast<size_t>(3 * out_h * out_w));
}

TEST_F(InferenceTest, PreprocessFromEmptyMat) {
    auto inf = make_inference();
    cv::Mat empty;
    int a = 0;
    int b = 0;
    int c = 0;
    int d = 0;
    EXPECT_THROW(inf->preprocess_image(empty, a, b, c, d), std::runtime_error);
}

TEST_F(InferenceTest, PreprocessFromMissingFile) {
    auto inf = make_inference();
    int a = 0;
    int b = 0;
    int c = 0;
    int d = 0;
    EXPECT_THROW(inf->preprocess_image(std::filesystem::path("/nonexistent.jpg"), a, b, c, d), std::runtime_error);
}

TEST_F(InferenceTest, PostprocessDepthShape) {
    const int out_h = 36;
    const int out_w = 48;
    const int orig_h = 100;
    const int orig_w = 200;

    auto inf = make_inference(out_h, out_w);

    // Simulate run_inference with a dummy tensor
    std::vector<float> dummy(static_cast<size_t>(3 * out_h * out_w), 0.0f);
    inf->run_inference(dummy, out_h, out_w);

    // Without upscaling
    auto result = inf->postprocess_outputs(out_h, out_w, orig_h, orig_w, false);
    EXPECT_EQ(result.depth.rows, out_h);
    EXPECT_EQ(result.depth.cols, out_w);

    // With upscaling
    inf->run_inference(dummy, out_h, out_w); // run again to repopulate cache
    auto result_up = inf->postprocess_outputs(out_h, out_w, orig_h, orig_w, true);
    EXPECT_EQ(result_up.depth.rows, orig_h);
    EXPECT_EQ(result_up.depth.cols, orig_w);
}

TEST_F(InferenceTest, PostprocessWithConf) {
    const int out_h = 36;
    const int out_w = 48;
    const int orig_h = 100;
    const int orig_w = 200;

    auto inf = make_inference(out_h, out_w, /*with_conf=*/true);

    std::vector<float> dummy(static_cast<size_t>(3 * out_h * out_w), 0.0f);
    inf->run_inference(dummy, out_h, out_w);

    auto result = inf->postprocess_outputs(out_h, out_w, orig_h, orig_w, false);
    EXPECT_FALSE(result.conf.empty());
    EXPECT_EQ(result.conf.rows, out_h);
    EXPECT_EQ(result.conf.cols, out_w);
}

TEST_F(InferenceTest, PostprocessWithoutRunning) {
    auto inf = make_inference();
    EXPECT_THROW(inf->postprocess_outputs(36, 48, 100, 200, false), std::runtime_error);
}

TEST_F(InferenceTest, ColorizerDepth) {
    cv::Mat depth(100, 100, CV_32F);
    cv::randu(depth, cv::Scalar(0.1f), cv::Scalar(10.0f));
    cv::Mat vis = DepthAnything3Inference::colorize_depth(depth);
    EXPECT_EQ(vis.rows, 100);
    EXPECT_EQ(vis.cols, 100);
    EXPECT_EQ(vis.channels(), 3);
}

// ============================================================================
// metric_depth_from_relative tests
// ============================================================================

TEST(MetricScaling, Formula) {
    // 500 * 1.5 / 300 = 2.5
    EXPECT_NEAR(da3::processing::metric_depth_from_relative(1.5f, 500.0f), 2.5f, 1e-5f);
    EXPECT_NEAR(da3::processing::metric_depth_from_relative(0.1f, 1000.0f), 100.0f / 300.0f, 1e-5f);
    EXPECT_NEAR(da3::processing::metric_depth_from_relative(1.0f, 300.0f), 1.0f, 1e-5f);
}

TEST(MetricScaling, ZeroInput) {
    EXPECT_NEAR(da3::processing::metric_depth_from_relative(0.0f, 500.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(da3::processing::metric_depth_from_relative(0.0f, 0.0f), 0.0f, 1e-5f);
}

TEST(MetricScaling, LargeFocalSanity) {
    // Large focal length scales proportionally: focal=5000, net=1.2 -> 20
    EXPECT_NEAR(da3::processing::metric_depth_from_relative(1.2f, 5000.0f), 20.0f, 1e-3f);
    constexpr float kHugeFocal = 1e6f;
    EXPECT_NEAR(da3::processing::metric_depth_from_relative(300.0f, kHugeFocal), kHugeFocal, kHugeFocal * 1e-5f);
}

// ============================================================================
// MockBackend direct tests
// ============================================================================

TEST(MockBackend, ReturnsConfiguredOutputs) {
    MockBackend backend;
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {1, 2, 2};
    backend.set_outputs({data}, {shape});

    EXPECT_EQ(backend.get_output_count(), 1u);
    EXPECT_EQ(backend.get_output_shape(0), shape);

    std::vector<float> out(4);
    backend.get_output_data(0, out.data(), 4);
    EXPECT_EQ(out, data);
}

TEST(MockBackend, OutOfRangeThrows) {
    MockBackend backend;
    backend.set_outputs({}, {});

    EXPECT_THROW(backend.get_output_shape(0), std::out_of_range);
    std::vector<float> out(4);
    EXPECT_THROW(backend.get_output_data(0, out.data(), 4), std::out_of_range);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
