#include "depth_anything3_inference.hpp"

#include "processing_utils.hpp"

#include <iostream>
#include <numeric>
#include <stdexcept>

DepthAnything3Inference::DepthAnything3Inference(const std::filesystem::path &model_path, const Config &config)
    : backend_(create_backend()), config_(config), input_shape_({1, 3, 0, 0}) {
    std::cout << "Using backend: " << backend_->get_backend_name() << std::endl;
    input_shape_ = backend_->initialize(model_path, input_shape_);

    // Validate at least one output (depth)
    const size_t num_outputs = backend_->get_output_count();
    if (num_outputs < 1) {
        throw std::runtime_error("Model must have at least 1 output (depth), but got " + std::to_string(num_outputs));
    }
    std::cout << "Model loaded successfully with " << num_outputs << " output(s)." << std::endl;
}

DepthAnything3Inference::DepthAnything3Inference(std::unique_ptr<InferenceBackend> backend, const Config &config)
    : backend_(std::move(backend)), config_(config), input_shape_({1, 3, 0, 0}) {}

std::vector<float> DepthAnything3Inference::preprocess_image(const std::filesystem::path &image_path, int &orig_h,
                                                             int &orig_w, int &out_h, int &out_w) {
    if (!std::filesystem::exists(image_path)) {
        throw std::runtime_error("Image file does not exist: " + image_path.string());
    }

    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not load image from: " + image_path.string());
    }

    return preprocess_image(image, orig_h, orig_w, out_h, out_w);
}

std::vector<float> DepthAnything3Inference::preprocess_image(const cv::Mat &bgr_image, int &orig_h, int &orig_w,
                                                             int &out_h, int &out_w) {
    if (bgr_image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    orig_h = bgr_image.rows;
    orig_w = bgr_image.cols;

    auto [h, w] = da3::processing::compute_output_size(orig_h, orig_w, config_.process_res, config_.patch_size);
    out_h = h;
    out_w = w;

    const auto num_pixels = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
    std::vector<float> input_tensor(3 * num_pixels);
    da3::processing::preprocess_frame(bgr_image, input_tensor, out_h, out_w, config_.means, config_.stds);
    return input_tensor;
}

void DepthAnything3Inference::run_inference(std::span<const float> input_data, int out_h, int out_w) {
    // Build the actual input shape for this call
    const std::vector<int64_t> shape{1, 3, static_cast<int64_t>(out_h), static_cast<int64_t>(out_w)};
    backend_->run_inference(input_data, shape);

    // Cache output tensors
    const size_t num_outputs = backend_->get_output_count();
    output_data_cache_.clear();
    output_shapes_cache_.clear();

    for (size_t i = 0; i < num_outputs; ++i) {
        auto out_shape = backend_->get_output_shape(i);
        const size_t size = std::accumulate(out_shape.begin(), out_shape.end(), size_t{1},
                                            [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });

        std::vector<float> data(size);
        backend_->get_output_data(i, data.data(), size);

        output_data_cache_.push_back(std::move(data));
        output_shapes_cache_.push_back(std::move(out_shape));
    }
}

void DepthAnything3Inference::run_inference_batch(const std::vector<std::vector<float>> &frames, int out_h, int out_w) {
    if (frames.empty()) {
        throw std::invalid_argument("run_inference_batch: no frames");
    }
    const auto frame_elems = static_cast<size_t>(3) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
    for (size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].size() < frame_elems) {
            throw std::invalid_argument("run_inference_batch: frame " + std::to_string(i) + " too small");
        }
    }
    std::vector<float> stacked;
    stacked.reserve(frames.size() * frame_elems);
    for (const auto &frame : frames) {
        stacked.insert(stacked.end(), frame.begin(), frame.begin() + static_cast<ptrdiff_t>(frame_elems));
    }
    const std::vector<int64_t> shape{static_cast<int64_t>(frames.size()), 3, static_cast<int64_t>(out_h),
                                     static_cast<int64_t>(out_w)};
    backend_->run_inference(stacked, shape);

    const size_t num_outputs = backend_->get_output_count();
    output_data_cache_.clear();
    output_shapes_cache_.clear();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto out_shape = backend_->get_output_shape(i);
        const size_t size = std::accumulate(out_shape.begin(), out_shape.end(), size_t{1},
                                            [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
        std::vector<float> data(size);
        backend_->get_output_data(i, data.data(), size);
        output_data_cache_.push_back(std::move(data));
        output_shapes_cache_.push_back(std::move(out_shape));
    }
}

namespace {

// Copy view @p view of an [N,H,W] (or [1,H,W]) float buffer into a CV_32F Mat.
// H/W come from the trailing shape dims, falling back to @p out_h/out_w.
cv::Mat extract_view_mat(const std::vector<float> &data, const std::vector<int64_t> &shape, size_t view, int out_h,
                         int out_w) {
    int h = out_h;
    int w = out_w;
    if (shape.size() >= 2) {
        h = static_cast<int>(shape[shape.size() - 2]);
        w = static_cast<int>(shape[shape.size() - 1]);
    }
    const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
    const size_t n_views = plane > 0 ? data.size() / plane : 1;
    const size_t v = (n_views > 1) ? view : 0;
    if ((v + 1) * plane > data.size()) {
        throw std::runtime_error("Output view " + std::to_string(view) + " out of range");
    }
    cv::Mat mat(h, w, CV_32F);
    const float *src = data.data() + v * plane;
    std::copy(src, src + plane, mat.ptr<float>());
    return mat;
}

} // namespace

DepthResult DepthAnything3Inference::postprocess_outputs(int out_h, int out_w, int orig_h, int orig_w,
                                                         bool upscale_to_orig) {
    if (output_data_cache_.empty()) {
        throw std::runtime_error("No outputs cached — call run_inference() first");
    }

    DepthResult result;
    result.orig_h = orig_h;
    result.orig_w = orig_w;

    // Output 0: depth map — shape may be [1, H, W], [H, W], [1, 1, H, W], or [1, N, 1, H, W]
    const auto &depth_data = output_data_cache_[0];
    const auto &depth_shape = output_shapes_cache_[0];

    // Determine spatial dimensions from the output shape (last two dims)
    int model_h = out_h;
    int model_w = out_w;
    if (depth_shape.size() >= 2) {
        model_h = static_cast<int>(depth_shape[depth_shape.size() - 2]);
        model_w = static_cast<int>(depth_shape[depth_shape.size() - 1]);
    }

    // Copy depth data into a cv::Mat (CV_32F)
    cv::Mat depth_mat(model_h, model_w, CV_32F);
    const size_t num_pixels = static_cast<size_t>(model_h) * static_cast<size_t>(model_w);
    if (depth_data.size() < num_pixels) {
        throw std::runtime_error("Depth output too small: expected " + std::to_string(num_pixels) + " pixels, got " +
                                 std::to_string(depth_data.size()));
    }
    // Take the last H*W elements to handle batch / channel leading dims
    const float *src = depth_data.data() + depth_data.size() - num_pixels;
    std::copy(src, src + num_pixels, depth_mat.ptr<float>());

    if (upscale_to_orig && (model_h != orig_h || model_w != orig_w)) {
        cv::resize(depth_mat, depth_mat, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
    }
    result.depth = std::move(depth_mat);

    // Output 1 (optional): confidence map
    if (config_.output_conf && output_data_cache_.size() >= 2) {
        const auto &conf_data = output_data_cache_[1];
        const auto &conf_shape = output_shapes_cache_[1];

        int conf_h = model_h;
        int conf_w = model_w;
        if (conf_shape.size() >= 2) {
            conf_h = static_cast<int>(conf_shape[conf_shape.size() - 2]);
            conf_w = static_cast<int>(conf_shape[conf_shape.size() - 1]);
        }

        const size_t conf_pixels = static_cast<size_t>(conf_h) * static_cast<size_t>(conf_w);
        cv::Mat conf_mat(conf_h, conf_w, CV_32F);
        if (conf_data.size() >= conf_pixels) {
            const float *conf_src = conf_data.data() + conf_data.size() - conf_pixels;
            std::copy(conf_src, conf_src + conf_pixels, conf_mat.ptr<float>());
        }

        if (upscale_to_orig && (conf_h != orig_h || conf_w != orig_w)) {
            cv::resize(conf_mat, conf_mat, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
        }
        result.conf = std::move(conf_mat);
    }

    return result;
}

DepthResult DepthAnything3Inference::postprocess_view(size_t view, int out_h, int out_w, int orig_h, int orig_w,
                                                      bool upscale_to_orig) {
    if (output_data_cache_.empty()) {
        throw std::runtime_error("No outputs cached — call run_inference_batch() first");
    }
    DepthResult result;
    result.orig_h = orig_h;
    result.orig_w = orig_w;

    cv::Mat depth_mat = extract_view_mat(output_data_cache_[0], output_shapes_cache_[0], view, out_h, out_w);
    if (upscale_to_orig && (depth_mat.rows != orig_h || depth_mat.cols != orig_w)) {
        cv::resize(depth_mat, depth_mat, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
    }
    result.depth = std::move(depth_mat);

    if (config_.output_conf && output_data_cache_.size() >= 2) {
        cv::Mat conf_mat = extract_view_mat(output_data_cache_[1], output_shapes_cache_[1], view, out_h, out_w);
        if (upscale_to_orig && (conf_mat.rows != orig_h || conf_mat.cols != orig_w)) {
            cv::resize(conf_mat, conf_mat, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
        }
        result.conf = std::move(conf_mat);
    }
    return result;
}

bool DepthAnything3Inference::get_pose_view(size_t view, std::array<float, 12> &pose,
                                            std::array<float, 4> &intrinsics) const {
    pose = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    bool found_pose = false;

    for (size_t i = 0; i < output_shapes_cache_.size(); ++i) {
        const auto &shape = output_shapes_cache_[i];
        if (shape.size() < 2 || shape[shape.size() - 2] != 3 || shape[shape.size() - 1] != 4) {
            continue;
        }
        // Extrinsics [..,3,4]: leading dims are batch/view dims.
        const auto &data = output_data_cache_[i];
        const size_t n = data.size() / 12;
        if (view < n && data.size() >= (view + 1) * 12) {
            for (size_t k = 0; k < 12; ++k) {
                pose[k] = data[view * 12 + k];
            }
            found_pose = true;
        }
        break; // first [..,3,4] head wins
    }

    for (size_t i = 0; i < output_shapes_cache_.size(); ++i) {
        const auto &shape = output_shapes_cache_[i];
        if (shape.size() < 2 || shape[shape.size() - 2] != 3 || shape[shape.size() - 1] != 3) {
            continue;
        }
        // Intrinsics K = [fx 0 cx; 0 fy cy; 0 0 1] -> decompose properly
        // (the old main.cpp path copied the first 4 floats, yielding fy=0).
        const auto &data = output_data_cache_[i];
        const size_t n = data.size() / 9;
        if (view < n && data.size() >= (view + 1) * 9) {
            const float *k = data.data() + view * 9;
            intrinsics = {k[0], k[4], k[2], k[5]};
        }
        break; // first [..,3,3] head wins
    }

    return found_pose;
}

cv::Mat DepthAnything3Inference::colorize_depth(const cv::Mat &depth_map) {
    return da3::processing::apply_depth_colormap(depth_map);
}

cv::Mat DepthAnything3Inference::colorize_conf(const cv::Mat &conf_map) {
    return da3::processing::apply_conf_colormap(conf_map);
}

std::optional<std::filesystem::path> DepthAnything3Inference::save_output_image(const cv::Mat &image,
                                                                                const std::filesystem::path &path) {
    if (cv::imwrite(path.string(), image)) {
        return path;
    }
    return std::nullopt;
}
