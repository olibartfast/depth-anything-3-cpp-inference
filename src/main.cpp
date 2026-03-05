#include "depth_anything3_inference.hpp"

#include <iostream>

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to_model> <path_to_image> [--conf] [--no-upscale] [--process-res N]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  path_to_model   ONNX model (.onnx) or TensorRT engine (.engine / .trt)" << std::endl;
        std::cerr << "  path_to_image   Input image (JPG, PNG, …)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --conf          Also save the confidence map as output_conf.jpg" << std::endl;
        std::cerr << "  --no-upscale    Keep depth map at model output resolution (no resize to original)"
                  << std::endl;
        std::cerr << "  --process-res N Resize the longest side to N before inference (default: 504)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Note: Backend (ONNX Runtime or TensorRT) is selected at compile time." << std::endl;
        std::cerr << "      Build with -DUSE_ONNX_RUNTIME=ON or -DUSE_TENSORRT=ON" << std::endl;
        return 1;
    }

    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path image_path = argv[2];

    // Parse optional arguments
    bool output_conf = false;
    bool upscale_to_orig = true;
    int process_res = 504;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--conf") {
            output_conf = true;
        } else if (arg == "--no-upscale") {
            upscale_to_orig = false;
        } else if (arg == "--process-res" && i + 1 < argc) {
            try {
                process_res = std::stoi(argv[++i]);
                if (process_res <= 0) {
                    throw std::invalid_argument("process_res must be positive");
                }
            } catch (const std::exception &e) {
                std::cerr << "Invalid value for --process-res: " << e.what() << std::endl;
                return 1;
            }
        }
    }

    try {
        Config config;
        config.process_res = process_res;
        config.output_conf = output_conf;

        DepthAnything3Inference inference(model_path, config);

        int orig_h = 0;
        int orig_w = 0;
        int out_h = 0;
        int out_w = 0;

        std::vector<float> input_data = inference.preprocess_image(image_path, orig_h, orig_w, out_h, out_w);
        std::cout << "Input image: " << orig_h << "x" << orig_w << std::endl;
        std::cout << "Model input: " << out_h << "x" << out_w << std::endl;

        inference.run_inference(input_data, out_h, out_w);

        const DepthResult result = inference.postprocess_outputs(out_h, out_w, orig_h, orig_w, upscale_to_orig);

        // Colorize and save depth map
        const cv::Mat depth_vis = DepthAnything3Inference::colorize_depth(result.depth);
        const std::filesystem::path depth_out = "output_depth.jpg";
        if (const auto saved = DepthAnything3Inference::save_output_image(depth_vis, depth_out)) {
            std::cout << "Depth map saved to: " << saved->string() << std::endl;
        } else {
            throw std::runtime_error("Could not save depth map to " + depth_out.string());
        }

        // Print depth statistics
        double min_depth = 0.0;
        double max_depth = 0.0;
        cv::minMaxLoc(result.depth, &min_depth, &max_depth);
        std::cout << "Depth range: [" << min_depth << ", " << max_depth << "]" << std::endl;

        // Optionally save confidence map
        if (output_conf && !result.conf.empty()) {
            const cv::Mat conf_vis = DepthAnything3Inference::colorize_conf(result.conf);
            const std::filesystem::path conf_out = "output_conf.jpg";
            if (const auto saved = DepthAnything3Inference::save_output_image(conf_vis, conf_out)) {
                std::cout << "Confidence map saved to: " << saved->string() << std::endl;
            }
        } else if (output_conf) {
            std::cout << "Note: Model did not produce a confidence output." << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
