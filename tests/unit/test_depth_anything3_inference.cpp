#include "depth_anything3_inference.hpp"
#include "fusion_eval.hpp"
#include "mock_backend.hpp"
#include "processing_utils.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

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
        0.1f, 0.2f, // channel 0
        0.3f, 0.4f, // channel 1
        0.5f, 0.6f  // channel 2
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
    cv::randu(depth, cv::Scalar(0.5), cv::Scalar(10.0));
    cv::Mat colored = da3::processing::apply_depth_colormap(depth);
    EXPECT_EQ(colored.rows, 100);
    EXPECT_EQ(colored.cols, 100);
    EXPECT_EQ(colored.channels(), 3);
}

TEST(ApplyDepthColormap, UniformDepth) {
    // All-same depth should not crash
    cv::Mat depth(50, 50, CV_32F, cv::Scalar(5.0));
    EXPECT_NO_THROW({ auto _ = da3::processing::apply_depth_colormap(depth); });
}

// ============================================================================
// DepthAnything3Inference — constructor & config tests
// ============================================================================

class InferenceTest : public ::testing::Test {
  protected:
    // Build an inference instance using a MockBackend with a single depth output
    std::unique_ptr<DepthAnything3Inference> make_inference(int out_h = 36, int out_w = 36, bool with_conf = false) {
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
    EXPECT_THROW({ (void)inf->postprocess_outputs(36, 48, 100, 200, false); }, std::runtime_error);
}

TEST_F(InferenceTest, ColorizerDepth) {
    cv::Mat depth(100, 100, CV_32F);
    cv::randu(depth, cv::Scalar(0.1), cv::Scalar(10.0));
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

    EXPECT_THROW({ (void)backend.get_output_shape(0); }, std::out_of_range);
    std::vector<float> out(4);
    EXPECT_THROW(backend.get_output_data(0, out.data(), 4), std::out_of_range);
}

// ============================================================================
// NPY raw-dump header layout
// ============================================================================

namespace {

// Mirrors write_npy_float32() in src/main.cpp. main.cpp is not includable from
// the test target, so the writer is duplicated here to assert the on-disk NPY
// v1.0 layout (magic, '<f4' descr, fortran_order False, shape tuple, 64-byte
// preamble alignment). Keep in sync with the helper in src/main.cpp.
void write_npy_float32_for_test(const cv::Mat &mat, const std::filesystem::path &path) {
    ASSERT_EQ(mat.type(), CV_32F);

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(mat.rows) + ", " +
                         std::to_string(mat.cols) + "), }";

    const size_t preamble = 6 + 2 + 2;
    const size_t unpadded = preamble + header.size() + 1;
    header.append((64 - (unpadded % 64)) % 64, ' ');
    header.push_back('\n');

    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out.is_open());

    const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    out.write(magic, sizeof(magic));
    out.put('\x01');
    out.put('\x00');
    const auto header_len = static_cast<uint16_t>(header.size());
    out.put(static_cast<char>(header_len & 0xFFu));
    out.put(static_cast<char>((header_len >> 8u) & 0xFFu));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    const auto row_bytes = static_cast<std::streamsize>(mat.cols) * static_cast<std::streamsize>(sizeof(float));
    for (int y = 0; y < mat.rows; ++y) {
        out.write(reinterpret_cast<const char *>(mat.ptr<float>(y)), row_bytes);
    }
}

} // namespace

TEST(NpyDump, HeaderLayoutMatchesNpyV1) {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "da3_npy_header_test.npy";
    cv::Mat depth(3, 2, CV_32F);
    depth.at<float>(0, 0) = 1.5f;
    depth.at<float>(2, 1) = -2.25f;

    write_npy_float32_for_test(depth, tmp);

    std::ifstream in(tmp, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    const std::vector<char> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    in.close();
    std::filesystem::remove(tmp);

    ASSERT_GE(bytes.size(), 10u);
    // Magic "\x93NUMPY"
    EXPECT_EQ(static_cast<unsigned char>(bytes[0]), 0x93u);
    EXPECT_EQ(bytes[1], 'N');
    EXPECT_EQ(bytes[2], 'U');
    EXPECT_EQ(bytes[3], 'M');
    EXPECT_EQ(bytes[4], 'P');
    EXPECT_EQ(bytes[5], 'Y');
    // Version 1.0
    EXPECT_EQ(static_cast<unsigned char>(bytes[6]), 0x01u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[7]), 0x00u);

    // Little-endian uint16 header length, then the python-dict header.
    const auto hlen = static_cast<unsigned>(static_cast<unsigned char>(bytes[8])) |
                      (static_cast<unsigned>(static_cast<unsigned char>(bytes[9])) << 8u);
    ASSERT_GE(bytes.size(), 10u + hlen);
    const std::string header(bytes.begin() + 10, bytes.begin() + 10 + hlen);

    EXPECT_NE(header.find("'descr': '<f4'"), std::string::npos);
    EXPECT_NE(header.find("'fortran_order': False"), std::string::npos);
    EXPECT_NE(header.find("'shape': (3, 2)"), std::string::npos);
    EXPECT_EQ(header.back(), '\n');
    // Preamble (magic + version + length + header) must be 64-byte aligned.
    EXPECT_EQ((10u + hlen) % 64u, 0u);
    // Payload: H * W little-endian float32 values.
    EXPECT_EQ(bytes.size(), static_cast<size_t>(10u + hlen) + 3u * 2u * sizeof(float));
}

// ============================================================================
// Native task CLI helpers (R-8): task parse, checkpoints, dir glob
// ============================================================================

TEST(TaskParse, RoundTrip) {
    EXPECT_EQ(parse_task(""), TaskType::mono);
    EXPECT_EQ(parse_task("mono"), TaskType::mono);
    EXPECT_EQ(parse_task("metric"), TaskType::metric);
    EXPECT_EQ(parse_task("multiview"), TaskType::multiview);
    EXPECT_EQ(parse_task("pose"), TaskType::pose);
    EXPECT_EQ(parse_task("reconstruction"), TaskType::reconstruction);
    EXPECT_EQ(parse_task("gaussian"), TaskType::gaussian);
    EXPECT_EQ(task_to_string(TaskType::mono), "mono");
    EXPECT_EQ(task_to_string(TaskType::metric), "metric");
    EXPECT_EQ(task_to_string(TaskType::multiview), "multiview");
    EXPECT_EQ(task_to_string(TaskType::pose), "pose");
    EXPECT_EQ(task_to_string(TaskType::reconstruction), "reconstruction");
    EXPECT_EQ(task_to_string(TaskType::gaussian), "gaussian");
    // Round-trip through the string form.
    for (const std::string name : {"mono", "metric", "multiview", "pose", "reconstruction", "gaussian"}) {
        EXPECT_EQ(task_to_string(parse_task(name)), name);
    }
}

TEST(TaskParse, InvalidThrows) {
    EXPECT_THROW({ (void)parse_task("slam"); }, std::runtime_error);
}

TEST(TaskCheckpoints, Recommended) {
    EXPECT_NE(recommended_checkpoint(TaskType::mono).find("SMALL"), std::string::npos);
    EXPECT_NE(recommended_checkpoint(TaskType::metric).find("METRIC"), std::string::npos);
    EXPECT_NE(recommended_checkpoint(TaskType::pose).find("not METRIC"), std::string::npos);
    EXPECT_NE(recommended_checkpoint(TaskType::reconstruction).find("NESTED"), std::string::npos);
    EXPECT_NE(recommended_checkpoint(TaskType::gaussian).find("Gaussian"), std::string::npos);
}

TEST(MetricTaskScaling, Formula) {
    // Same contract as MetricScaling, routed as the --task metric path uses it:
    // metric = focal * net / 300, applied per pixel.
    const float focal = 500.0f;
    const float net = 1.5f;
    EXPECT_NEAR(da3::processing::metric_depth_from_relative(net, focal), focal * net / 300.0f, 1e-5f);
}

TEST(CollectImages, SortedGlob) {
    const auto dir = std::filesystem::temp_directory_path() / "da3_collect_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "b.jpg").put('x');
    std::ofstream(dir / "a.png").put('x');
    std::ofstream(dir / "c.txt").put('x');
    const auto paths = collect_image_paths(dir);
    ASSERT_EQ(paths.size(), 2u);
    EXPECT_EQ(paths[0].filename().string(), "a.png");
    EXPECT_EQ(paths[1].filename().string(), "b.jpg");
    EXPECT_TRUE(collect_image_paths(dir / "missing").empty());
    std::filesystem::remove_all(dir);
}

// ============================================================================
// Pose / PLY / Gaussian file formats — exercise the REAL writers in
// da3::tasks (src/depth_anything3_inference.hpp), the same code main.cpp uses.
// ============================================================================

TEST(PoseTxt, Format) {
    const auto tmp = std::filesystem::temp_directory_path() / "da3_poses_test.txt";
    const std::vector<std::array<float, 12>> poses = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
                                                      {1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3}};
    ASSERT_TRUE(tasks::write_poses_txt(tmp, poses));
    std::ifstream in(tmp);
    ASSERT_TRUE(in.is_open());
    std::string line;
    int n_lines = 0;
    while (std::getline(in, line)) {
        ++n_lines;
        std::istringstream iss(line);
        const std::vector<float> vals{std::istream_iterator<float>(iss), std::istream_iterator<float>()};
        EXPECT_EQ(vals.size(), 12u); // N x 12 floats (row-major 3x4 w2c)
    }
    EXPECT_EQ(n_lines, 2);
    std::filesystem::remove(tmp);
}

TEST(PoseIntrinsics, DefaultFocal) {
    // R-5: default focal is 1.2 * width when --focal is not passed.
    EXPECT_NEAR(tasks::default_focal(500), 600.0f, 1e-5f);
    EXPECT_NEAR(tasks::default_focal(100), 120.0f, 1e-5f);
}

namespace {
// Build a real FrameData for the PLY writer tests.
tasks::FrameData make_test_frame(int rows, int cols, float depth_value) {
    tasks::FrameData frame;
    frame.depth = cv::Mat(rows, cols, CV_32F, cv::Scalar(static_cast<double>(depth_value)));
    frame.image_bgr = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(10, 20, 30));
    frame.pose = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    const float fx = tasks::default_focal(cols);
    frame.intrinsics = {fx, fx, static_cast<float>(cols) / 2.0f, static_cast<float>(rows) / 2.0f};
    return frame;
}
} // namespace

TEST(PlyHeader, WritesVertices) {
    const auto tmp = std::filesystem::temp_directory_path() / "da3_cloud_test.ply";
    const std::vector<tasks::FrameData> frames = {make_test_frame(2, 3, 2.0f)};
    ASSERT_TRUE(tasks::save_point_cloud_ply(tmp, frames));
    std::ifstream in(tmp);
    ASSERT_TRUE(in.is_open());
    const std::string content{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    EXPECT_NE(content.find("ply"), std::string::npos);
    EXPECT_NE(content.find("element vertex 6"), std::string::npos); // H*W*N = 2*3*1
    EXPECT_NE(content.find("property float x"), std::string::npos);
    EXPECT_NE(content.find("property uchar red"), std::string::npos);
    std::filesystem::remove(tmp);
}

TEST(GaussianPly, NonEmpty) {
    const auto tmp = std::filesystem::temp_directory_path() / "da3_gauss_test.ply";
    const std::vector<tasks::FrameData> frames = {make_test_frame(2, 2, 1.5f)};
    ASSERT_TRUE(tasks::save_gaussians_ply(tmp, frames));
    EXPECT_GT(std::filesystem::file_size(tmp), 0u);
    std::ifstream in(tmp);
    const std::string content{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    EXPECT_NE(content.find("element vertex 4"), std::string::npos);
    EXPECT_NE(content.find("opacity"), std::string::npos);
    EXPECT_NE(content.find("scale_0"), std::string::npos);
    std::filesystem::remove(tmp);
}

// ============================================================================
// Camera presets (R-4) — call the production header functions directly.
// ============================================================================

TEST(CameraParse, RoundTrip) {
    EXPECT_EQ(parse_camera(""), CameraPreset::none);
    EXPECT_EQ(parse_camera("d405"), CameraPreset::d405);
    EXPECT_EQ(parse_camera("d435"), CameraPreset::d435);
    EXPECT_EQ(parse_camera("l515"), CameraPreset::l515);
    EXPECT_EQ(parse_camera("kinect"), CameraPreset::kinect);
    EXPECT_EQ(parse_camera("zed2i-neural"), CameraPreset::zed2i_neural);
    EXPECT_EQ(parse_camera("zed2i-quality"), CameraPreset::zed2i_quality);
    EXPECT_EQ(parse_camera("base"), CameraPreset::base);
    EXPECT_EQ(camera_to_string(CameraPreset::d405), "d405");
    EXPECT_EQ(camera_to_string(CameraPreset::zed2i_neural), "zed2i-neural");
    EXPECT_EQ(camera_to_string(CameraPreset::zed2i_quality), "zed2i-quality");
    EXPECT_EQ(camera_to_string(CameraPreset::base), "base");
    EXPECT_EQ(camera_to_string(CameraPreset::none), "");
}

TEST(CameraParse, InvalidThrows) {
    EXPECT_THROW({ (void)parse_camera("realsense"); }, std::runtime_error);
}

TEST(CameraCheckpoint, Mapping) {
    EXPECT_EQ(camera_checkpoint(CameraPreset::d405), "d405");
    EXPECT_EQ(camera_checkpoint(CameraPreset::d435), "d435");
    EXPECT_EQ(camera_checkpoint(CameraPreset::base), "base");
    EXPECT_EQ(camera_checkpoint(CameraPreset::none), "");
    // Verified against the HF collection page (2026-10-13).
    EXPECT_EQ(camera_hf_id(CameraPreset::d435), "depth-anything/camera-depth-model-d435");
    EXPECT_EQ(camera_hf_id(CameraPreset::base), "depth-anything/camera-depth-model-base");
    EXPECT_EQ(camera_hf_id(CameraPreset::none), "");
}

TEST(CameraFocalPrecedence, ExplicitFocalWins) {
    // base implies already-metric only when no explicit --focal was passed.
    EXPECT_TRUE(camera_implies_already_metric(CameraPreset::base, /*focal_set=*/false));
    EXPECT_FALSE(camera_implies_already_metric(CameraPreset::base, /*focal_set=*/true));
    EXPECT_FALSE(camera_implies_already_metric(CameraPreset::d435, false));
    EXPECT_FALSE(camera_implies_already_metric(CameraPreset::none, false));
}

// ============================================================================
// N-view batch API (Phase B): run_inference_batch + postprocess_view +
// get_pose_view against a mock [N,H,W] / [N,3,4] / [N,3,3] backend.
// ============================================================================

namespace {

std::unique_ptr<DepthAnything3Inference> make_batch_inference() {
    Config cfg;
    cfg.output_conf = true;
    // 2 views, 2x3 spatial: depth values 1..12 (view0: 1..6, view1: 7..12).
    std::vector<float> depth(12);
    std::iota(depth.begin(), depth.end(), 1.0f);
    std::vector<float> conf(12, 0.5f);
    // Extrinsics: view1 carries tx=5; view0 identity.
    std::vector<float> extr = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0};
    // Intrinsics K = [fx 0 cx; 0 fy cy; 0 0 1] with fx=100, fy=200, cx=3, cy=4.
    std::vector<float> intr = {100, 0, 3, 0, 200, 4, 0, 0, 1, 100, 0, 3, 0, 200, 4, 0, 0, 1};
    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs({depth, conf, extr, intr}, {{2, 2, 3}, {2, 2, 3}, {2, 3, 4}, {2, 3, 3}});
    return std::make_unique<DepthAnything3Inference>(std::move(backend), cfg);
}

} // namespace

TEST(BatchInference, RunAndSliceViews) {
    auto inf = make_batch_inference();
    const std::vector<float> frame(static_cast<size_t>(3 * 2 * 3), 0.0f);
    inf->run_inference_batch({frame, frame}, 2, 3);
    auto v0 = inf->postprocess_view(0, 2, 3, 2, 3, false);
    auto v1 = inf->postprocess_view(1, 2, 3, 2, 3, false);
    EXPECT_FLOAT_EQ(v0.depth.at<float>(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(v0.depth.at<float>(1, 2), 6.0f);
    EXPECT_FLOAT_EQ(v1.depth.at<float>(0, 0), 7.0f);
    EXPECT_FLOAT_EQ(v1.depth.at<float>(1, 2), 12.0f);
    EXPECT_FALSE(v0.conf.empty());
    EXPECT_THROW({ (void)inf->postprocess_view(2, 2, 3, 2, 3, false); }, std::runtime_error);
}

TEST(BatchInference, EmptyThrows) {
    auto inf = make_batch_inference();
    EXPECT_THROW(inf->run_inference_batch({}, 2, 3), std::invalid_argument);
    EXPECT_THROW({ (void)inf->postprocess_view(0, 2, 3, 2, 3, false); }, std::runtime_error);
}

TEST(BatchPoseView, PerViewExtrinsicsAndK) {
    auto inf = make_batch_inference();
    const std::vector<float> frame(static_cast<size_t>(3 * 2 * 3), 0.0f);
    inf->run_inference_batch({frame, frame}, 2, 3);
    std::array<float, 12> pose{};
    std::array<float, 4> intr{};
    EXPECT_TRUE(inf->get_pose_view(0, pose, intr));
    EXPECT_FLOAT_EQ(pose[3], 0.0f); // view0 tx
    // Proper K decomposition: {fx, fy, cx, cy} = {100, 200, 3, 4}.
    EXPECT_FLOAT_EQ(intr[0], 100.0f);
    EXPECT_FLOAT_EQ(intr[1], 200.0f);
    EXPECT_FLOAT_EQ(intr[2], 3.0f);
    EXPECT_FLOAT_EQ(intr[3], 4.0f);
    EXPECT_TRUE(inf->get_pose_view(1, pose, intr));
    EXPECT_FLOAT_EQ(pose[3], 5.0f); // view1 tx
}

TEST(BatchPoseView, IdentityWithoutHead) {
    Config cfg;
    auto backend = std::make_unique<MockBackend>();
    backend->set_outputs({{1.0f, 2.0f, 3.0f, 4.0f}}, {{1, 2, 2}});
    DepthAnything3Inference inf(std::move(backend), cfg);
    const std::vector<float> frame(static_cast<size_t>(3 * 2 * 2), 0.0f);
    inf.run_inference(frame, 2, 2);
    std::array<float, 12> pose{};
    std::array<float, 4> intr{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_FALSE(inf.get_pose_view(0, pose, intr));
    EXPECT_FLOAT_EQ(pose[0], 1.0f); // identity fallback
    EXPECT_FLOAT_EQ(intr[0], 1.0f); // untouched without a head
}

TEST(PlyStride, Subsamples) {
    tasks::FrameData frame;
    frame.depth = cv::Mat(4, 4, CV_32F, cv::Scalar(2.0));
    frame.image_bgr = cv::Mat(4, 4, CV_8UC3, cv::Scalar(1, 2, 3));
    frame.pose = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    frame.intrinsics = {100, 100, 2, 2};
    const auto tmp = std::filesystem::temp_directory_path() / "da3_stride_test.ply";
    ASSERT_TRUE(tasks::save_point_cloud_ply(tmp, {frame}, 2));
    std::ifstream in(tmp);
    std::string content{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    EXPECT_NE(content.find("element vertex 4"), std::string::npos); // 2x2 of 4x4
    std::filesystem::remove(tmp);
}

TEST(VideoExtract, SamplesByFps) {
    const auto dir = std::filesystem::temp_directory_path() / "da3_video_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto avi = dir / "t.avi";
    cv::Mat frame(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::VideoWriter writer(avi.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(64, 64));
    ASSERT_TRUE(writer.isOpened());
    for (int i = 0; i < 5; ++i) {
        writer.write(frame);
    }
    writer.release();
    const auto out = dir / "frames";
    // 5 frames @10fps sampled at 5fps (step 2) -> frames 0, 2, 4.
    EXPECT_EQ(tasks::extract_video_frames(avi, 5.0, out), 3u);
    EXPECT_EQ(tasks::extract_video_frames(avi, 0.0, dir / "all"), 5u);
    EXPECT_THROW({ (void)tasks::extract_video_frames(dir / "missing.avi", 0.0, out); }, std::runtime_error);
    std::filesystem::remove_all(dir);
}

// ============================================================================
// Fusion preprocessing helpers (src/fusion_utils.hpp)
// ============================================================================

TEST(FusionPreprocess, ScaleIntrinsics) {
    const da3::fusion::Pinhole k{500.0F, 400.0F, 320.0F, 240.0F};
    const da3::fusion::Pinhole scaled = da3::fusion::scale_intrinsics(k, 640, 480, 320, 240);
    EXPECT_FLOAT_EQ(scaled.fx, 250.0F);
    EXPECT_FLOAT_EQ(scaled.fy, 200.0F);
    EXPECT_FLOAT_EQ(scaled.cx, 160.0F);
    EXPECT_FLOAT_EQ(scaled.cy, 120.0F);

    // A non-positive source size is a no-op rather than a division by zero.
    const da3::fusion::Pinhole unchanged = da3::fusion::scale_intrinsics(k, 0, 0, 320, 240);
    EXPECT_FLOAT_EQ(unchanged.fx, k.fx);
    EXPECT_FLOAT_EQ(unchanged.fy, k.fy);
    EXPECT_FLOAT_EQ(unchanged.cx, k.cx);
    EXPECT_FLOAT_EQ(unchanged.cy, k.cy);
}

TEST(FusionPreprocess, MaskLowConfidence) {
    const cv::Mat depth(4, 4, CV_32F, cv::Scalar(2.0));
    cv::Mat conf(4, 4, CV_32F, cv::Scalar(0.9));
    conf.at<float>(1, 2) = 0.2F; // below threshold -> zeroed
    conf.at<float>(3, 3) = 0.5F; // exactly at threshold -> kept

    const cv::Mat masked = da3::fusion::mask_low_confidence(depth, conf, 0.5F);
    EXPECT_FLOAT_EQ(masked.at<float>(1, 2), 0.0F);
    EXPECT_FLOAT_EQ(masked.at<float>(3, 3), 2.0F);
    EXPECT_FLOAT_EQ(masked.at<float>(0, 0), 2.0F);
    // The input depth is not modified.
    EXPECT_FLOAT_EQ(depth.at<float>(1, 2), 2.0F);

    // Empty confidence means "no masking": depth is returned unchanged.
    const cv::Mat unmasked = da3::fusion::mask_low_confidence(depth, cv::Mat(), 0.5F);
    for (int y = 0; y < unmasked.rows; ++y) {
        for (int x = 0; x < unmasked.cols; ++x) {
            EXPECT_FLOAT_EQ(unmasked.at<float>(y, x), 2.0F);
        }
    }
}

TEST(FusionPreprocess, ComputeValidMask) {
    cv::Mat depth(4, 4, CV_32F, cv::Scalar(2.0));
    depth.at<float>(2, 2) = 0.0F; // invalid depth -> invalid mask

    const cv::Mat valid = da3::fusion::compute_valid_mask(depth, 0.3F);
    EXPECT_EQ(valid.type(), CV_8U);
    EXPECT_EQ(valid.at<std::uint8_t>(2, 2), 0);
    EXPECT_EQ(valid.at<std::uint8_t>(0, 0), 255);
    EXPECT_EQ(valid.at<std::uint8_t>(0, 3), 255);
    EXPECT_EQ(valid.at<std::uint8_t>(3, 0), 255);

    // With edge_thresh <= 0, only non-positive depths are culled.
    cv::Mat flat(4, 4, CV_32F, cv::Scalar(1.5));
    flat.at<float>(0, 0) = -1.0F;
    flat.at<float>(3, 3) = 0.0F;
    const cv::Mat flat_valid = da3::fusion::compute_valid_mask(flat, 0.0F);
    EXPECT_EQ(flat_valid.at<std::uint8_t>(0, 0), 0);
    EXPECT_EQ(flat_valid.at<std::uint8_t>(3, 3), 0);
    EXPECT_EQ(flat_valid.at<std::uint8_t>(1, 1), 255);
}

TEST(FusionPreprocess, EstimateScaleMedian) {
    const cv::Mat pred(4, 4, CV_32F, cv::Scalar(2.0));
    cv::Mat ref(4, 4, CV_32F, cv::Scalar(5.0));
    EXPECT_DOUBLE_EQ(da3::fusion::estimate_scale_median(pred, ref, cv::Mat()), 2.5);

    // An outlier is excluded by the valid mask.
    ref.at<float>(0, 0) = 100.0F;
    cv::Mat valid(4, 4, CV_8U, cv::Scalar(255));
    valid.at<std::uint8_t>(0, 0) = 0;
    EXPECT_DOUBLE_EQ(da3::fusion::estimate_scale_median(pred, ref, valid), 2.5);

    // Nothing valid, or no positive ratios, falls back to 1.0.
    const cv::Mat none(4, 4, CV_8U, cv::Scalar(0));
    EXPECT_DOUBLE_EQ(da3::fusion::estimate_scale_median(pred, ref, none), 1.0);
    const cv::Mat zeros(4, 4, CV_32F, cv::Scalar(0.0));
    EXPECT_DOUBLE_EQ(da3::fusion::estimate_scale_median(zeros, ref, cv::Mat()), 1.0);
}

TEST(FusionPreprocess, InvertPoseW2cToC2w) {
    // R = 90 deg about z, t = (1, 2, 3).
    const std::array<float, 12> w2c = {0.0F, -1.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 1.0F, 3.0F};
    const std::array<float, 12> c2w = da3::fusion::invert_pose_w2c_to_c2w(w2c);

    // R^T = [0 1 0; -1 0 0; 0 0 1], t' = -R^T t = (-2, 1, -3).
    EXPECT_NEAR(c2w[0], 0.0F, 1e-6F);
    EXPECT_NEAR(c2w[1], 1.0F, 1e-6F);
    EXPECT_NEAR(c2w[3], -2.0F, 1e-6F);
    EXPECT_NEAR(c2w[4], -1.0F, 1e-6F);
    EXPECT_NEAR(c2w[7], 1.0F, 1e-6F);
    EXPECT_NEAR(c2w[11], -3.0F, 1e-6F);

    // Inverting twice returns the original w2c pose.
    const std::array<float, 12> round_trip = da3::fusion::invert_pose_w2c_to_c2w(c2w);
    for (std::size_t i = 0; i < w2c.size(); ++i) {
        EXPECT_NEAR(round_trip[i], w2c[i], 1e-6F) << "element " << i;
    }
}

// ============================================================================
// Fusion metrics (src/fusion_eval.hpp) on synthetic 4x4 mats/point grids.
// ============================================================================

TEST(FusionEval, AbsRel) {
    const cv::Mat ref(4, 4, CV_32F, cv::Scalar(1.0));
    const cv::Mat pred(4, 4, CV_32F, cv::Scalar(1.5));
    // All 16 pixels are within [0.3, 3.0]: |1.5 - 1.0| / 1.0 = 0.5.
    EXPECT_DOUBLE_EQ(da3::fusion::eval_abs_rel(pred, ref), 0.5);
    // The wrapper pins the default metric range to 0.3 m .. 3.0 m.
    EXPECT_DOUBLE_EQ(da3::fusion::eval_abs_rel(pred, ref), da3::fusion::abs_rel(pred, ref, cv::Mat(), 0.3, 3.0));

    // Out-of-range references and non-positive predictions are skipped.
    cv::Mat ref_sparse = ref.clone();
    cv::Mat pred_sparse = pred.clone();
    ref_sparse.at<float>(0, 0) = 5.0F;
    pred_sparse.at<float>(0, 0) = 5.0F;
    pred_sparse.at<float>(0, 1) = 0.0F;
    EXPECT_DOUBLE_EQ(da3::fusion::eval_abs_rel(pred_sparse, ref_sparse), 0.5);

    // An empty valid mask means all pixels; an all-zero mask means none (0.0).
    const cv::Mat mask(4, 4, CV_8U, cv::Scalar(0));
    EXPECT_DOUBLE_EQ(da3::fusion::eval_abs_rel(pred, ref, mask), 0.0);
}

TEST(FusionEval, AteTranslationRmse) {
    const std::array<float, 12> identity = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    std::array<float, 12> moved = identity;
    moved[3] = 3.0F; // translation only along x

    std::vector<std::array<float, 12>> est = {identity, moved};
    std::vector<std::array<float, 12>> ref = {identity, identity};
    // sqrt((0^2 + 3^2) / 2) = 3 / sqrt(2).
    EXPECT_NEAR(da3::fusion::eval_ate_translation_rmse(est, ref), 3.0 / std::sqrt(2.0), 1e-6);
    EXPECT_DOUBLE_EQ(da3::fusion::eval_ate_translation_rmse(est, ref), da3::fusion::ate_translation_rmse(est, ref));

    // Size mismatch and empty input use the -1.0 sentinel.
    const std::vector<std::array<float, 12>> one = {identity};
    const std::vector<std::array<float, 12>> none;
    EXPECT_DOUBLE_EQ(da3::fusion::eval_ate_translation_rmse(one, ref), -1.0);
    EXPECT_DOUBLE_EQ(da3::fusion::eval_ate_translation_rmse(none, none), -1.0);
}

TEST(FusionEval, ChamferL1) {
    using Points = std::vector<std::array<float, 3>>;
    const Points empty;
    const Points origin = {{0.0F, 0.0F, 0.0F}};
    const Points diagonal = {{1.0F, 1.0F, 0.0F}};

    EXPECT_DOUBLE_EQ(da3::fusion::chamfer_l1(empty, origin), -1.0);
    EXPECT_DOUBLE_EQ(da3::fusion::chamfer_l1(origin, empty), -1.0);

    // Manhattan distance: |1| + |1| + |0| = 2, not the Euclidean sqrt(2).
    EXPECT_DOUBLE_EQ(da3::fusion::chamfer_l1(origin, diagonal), 2.0);
    EXPECT_DOUBLE_EQ(da3::fusion::chamfer_l1(diagonal, origin), 2.0);
    EXPECT_DOUBLE_EQ(da3::fusion::chamfer_l1(origin, origin), 0.0);

    // Unequal set sizes: a->b = (1 + 2 + 6) / 3 = 3, b->a = (1 + 2) / 2 = 1.5.
    const Points a = {{0.0F, 0.0F, 0.0F}, {4.0F, 0.0F, 0.0F}, {8.0F, 0.0F, 0.0F}};
    const Points b = {{1.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}};
    EXPECT_DOUBLE_EQ(da3::fusion::chamfer_l1(a, b), 2.25);
    EXPECT_DOUBLE_EQ(da3::fusion::chamfer_l1(a, b), da3::fusion::chamfer_l1(b, a));

    // Synthetic 4x4 point grid vs the same plane lifted 1 m in z.
    Points plane;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            plane.push_back({static_cast<float>(x), static_cast<float>(y), 0.0F});
        }
    }
    Points lifted = plane;
    for (std::array<float, 3> &p : lifted) {
        p[2] = 1.0F;
    }
    EXPECT_DOUBLE_EQ(da3::fusion::chamfer_l1(plane, lifted), 1.0);
    EXPECT_DOUBLE_EQ(da3::fusion::chamfer_l1(plane, lifted), da3::fusion::chamfer_l1(lifted, plane));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
