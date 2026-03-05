#include "processing_utils.hpp"

#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>

static void BM_ComputeOutputSize(benchmark::State &state) {
    for (auto _ : state) {
        auto result = da3::processing::compute_output_size(1080, 1920, 504, 14);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ComputeOutputSize);

static void BM_PreprocessFrame(benchmark::State &state) {
    const int in_h = 480;
    const int in_w = 640;
    cv::Mat img(in_h, in_w, CV_8UC3, cv::Scalar(128, 100, 80));

    auto [out_h, out_w] = da3::processing::compute_output_size(in_h, in_w, 504, 14);
    std::vector<float> tensor(static_cast<size_t>(3 * out_h * out_w));
    std::array<float, 3> means = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds = {0.229f, 0.224f, 0.225f};

    for (auto _ : state) {
        da3::processing::preprocess_frame(img, tensor, out_h, out_w, means, stds);
        benchmark::DoNotOptimize(tensor.data());
    }
}
BENCHMARK(BM_PreprocessFrame);

static void BM_ApplyDepthColormap(benchmark::State &state) {
    cv::Mat depth(504, 504, CV_32F);
    cv::randu(depth, cv::Scalar(0.1f), cv::Scalar(10.0f));

    for (auto _ : state) {
        auto result = da3::processing::apply_depth_colormap(depth);
        benchmark::DoNotOptimize(result.data);
    }
}
BENCHMARK(BM_ApplyDepthColormap);

BENCHMARK_MAIN();
