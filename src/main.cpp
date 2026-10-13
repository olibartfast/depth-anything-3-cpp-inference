#include "depth_anything3_inference.hpp"
#include "processing_utils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Raw depth dump helpers
// ---------------------------------------------------------------------------

// Write a CV_32F (H, W) matrix as a raw float32 .npy file (NPY v1.0).
// Hand-rolled to avoid any new dependency; rows are written individually so
// padded (non-continuous) Mats are handled correctly. Returns false on I/O
// failure.
static bool write_npy_float32(const cv::Mat &mat, const std::filesystem::path &path) {
    CV_Assert(mat.type() == CV_32F);

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(mat.rows) + ", " +
                         std::to_string(mat.cols) + "), }";

    // NPY v1.0: total size of magic(6) + version(2) + header-len(2) + header
    // must be a multiple of 64 bytes and the header ends with '\n'.
    const size_t preamble = 6 + 2 + 2;
    const size_t unpadded = preamble + header.size() + 1;
    header.append((64 - (unpadded % 64)) % 64, ' ');
    header.push_back('\n');

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    out.write(magic, sizeof(magic));
    out.put('\x01'); // major version
    out.put('\x00'); // minor version
    const auto header_len = static_cast<uint16_t>(header.size());
    out.put(static_cast<char>(header_len & 0xFFu));
    out.put(static_cast<char>((header_len >> 8u) & 0xFFu));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    const auto row_bytes = static_cast<std::streamsize>(mat.cols) * static_cast<std::streamsize>(sizeof(float));
    for (int y = 0; y < mat.rows; ++y) {
        out.write(reinterpret_cast<const char *>(mat.ptr<float>(y)), row_bytes);
    }

    return static_cast<bool>(out);
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

static std::string zero_padded(int index) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%03d", index);
    return std::string(buf);
}

// Insert a zero-padded frame index before the extension: name.ext -> name_000.ext.
static std::filesystem::path append_index(const std::filesystem::path &path, int index,
                                          const std::string &default_ext) {
    std::string ext = path.extension().string();
    if (ext.empty()) {
        ext = default_ext;
    }
    return path.parent_path() / (path.stem().string() + "_" + zero_padded(index) + ext);
}

// Join a filename with an optional output directory (empty -> current directory).
static std::filesystem::path join_output(const std::filesystem::path &output_dir, const std::string &name) {
    if (output_dir.empty()) {
        return name;
    }
    return output_dir / name;
}

// Per-frame raw .npy path for multi-view runs. A directory argument receives
// depth_000.npy …; any other path is treated as a filename/prefix.
static std::filesystem::path raw_frame_path(const std::filesystem::path &dump_raw, int index) {
    if (std::filesystem::is_directory(dump_raw)) {
        return dump_raw / ("depth_" + zero_padded(index) + ".npy");
    }
    return append_index(dump_raw, index, ".npy");
}

// ---------------------------------------------------------------------------
// Metric scaling
// ---------------------------------------------------------------------------

// Apply metric = focal_px * net / 300 in place. Nested models already output
// meters, so callers pass already_metric=true to skip scaling.
static void apply_metric_scaling(cv::Mat &depth, float focal_px, bool already_metric) {
    if (already_metric) {
        return;
    }
    for (int y = 0; y < depth.rows; ++y) {
        float *row = depth.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x) {
            row[x] = da3::processing::metric_depth_from_relative(row[x], focal_px);
        }
    }
}

// Per-frame accumulation and file writers live in da3::tasks
// (src/depth_anything3_inference.hpp) so unit tests exercise the real code.
using tasks::FrameData;

// Locate the first cached model output whose trailing dimensions are exactly
// (rows, cols) — used to find a [3,4] extrinsics or [3,3] intrinsics head.
// Shape-heuristic (not name-based): the backend interface exposes output
// indices + shapes, not names (see requirements R-4 note).
static const std::vector<float> *find_matrix_output(const DepthAnything3Inference &inference, int64_t rows,
                                                    int64_t cols) {
    const auto &shapes = inference.get_raw_output_shapes();
    const auto &data = inference.get_raw_outputs();
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        const auto &shape = shapes[i];
        if (shape.size() >= 2 && shape[shape.size() - 2] == rows && shape[shape.size() - 1] == cols) {
            return &data[i];
        }
    }
    return nullptr;
}

// Populate @p pose (always; identity fallback) and @p intrinsics (overridden
// only when the model exposes them) from the cached pose-head outputs.
static bool apply_model_pose(const DepthAnything3Inference &inference, std::array<float, 12> &pose,
                             std::array<float, 4> &intrinsics) {
    pose = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    bool found_pose = false;
    const std::vector<float> *extrinsics = find_matrix_output(inference, 3, 4);
    if (extrinsics != nullptr && extrinsics->size() >= 12) {
        for (std::size_t k = 0; k < 12; ++k) {
            pose[k] = (*extrinsics)[k];
        }
        found_pose = true;
    }

    const std::vector<float> *kin = find_matrix_output(inference, 3, 3);
    if (kin != nullptr && kin->size() >= 4) {
        for (std::size_t k = 0; k < 4; ++k) {
            intrinsics[k] = (*kin)[k];
        }
    }

    return found_pose;
}

// ---------------------------------------------------------------------------
// CLI options & usage
// ---------------------------------------------------------------------------

struct Options {
    TaskType task{TaskType::mono};
    CameraPreset camera{CameraPreset::none}; // --camera preset; none -> unset
    float focal{500.0f};
    bool focal_set{false}; // true when --focal was passed; else 1.2*W default
    std::filesystem::path input_dir;  // overrides the positional <image> when set
    std::filesystem::path output_dir; // empty -> current directory
    bool already_metric{false};
    bool use_ray_pose{false};
    bool output_conf{false};
    bool upscale_to_orig{true};
    int process_res{504};
    std::filesystem::path dump_raw;

    /// True when metric scaling must be skipped: either the user passed
    /// --already-metric, or the camera preset implies meters on its own (the
    /// Nested-family `base` checkpoint). An explicit --focal always wins.
    [[nodiscard]] bool skip_metric_scaling() const noexcept {
        return already_metric || camera_implies_already_metric(camera, focal_set);
    }
};

static void print_usage(std::ostream &os, const char *program) {
    os << "Usage: " << program << " <path_to_model> <path_to_image|dir> [options]\n";
    os << "\n";
    os << "Arguments:\n";
    os << "  path_to_model   ONNX model (.onnx) or TensorRT engine (.engine / .trt)\n";
    os << "  path_to_image   Input image (JPG, PNG, …) or directory of images\n";
    os << "\n";
    os << "Options:\n";
    os << "  --task TASK       Native DA3 task: mono, metric, multiview, pose,\n";
    os << "                    reconstruction, gaussian (default: mono)\n";
    os << "  --camera ID       Camera-depth preset: d405, d435, l515, kinect,\n";
    os << "                    zed2i-neural, zed2i-quality, base (default: none).\n";
    os << "                    Selects the recommended checkpoint under the HF\n";
    os << "                    collection depth-anything/camera-depth-models;\n";
    os << "                    camera models are metric-oriented (prefer --task\n";
    os << "                    metric). See the camera table below.\n";
    os << "  --focal PX        Focal length in pixels for metric scaling (default: 500;\n";
    os << "                    pose/reconstruction default to 1.2*W unless passed)\n";
    os << "  --input-dir DIR   Directory of input images (overrides positional image)\n";
    os << "  --output-dir DIR  Directory for output files (default: current directory)\n";
    os << "  --already-metric  Depth is already in meters; skip focal*net/300 scaling\n";
    os << "  --use-ray-pose    Use ray-based pose estimation for the pose task\n";
    os << "  --conf            Also save the confidence map as output_conf.jpg\n";
    os << "  --no-upscale      Keep depth map at model output resolution (no resize)\n";
    os << "  --process-res N   Resize the longest side to N before inference (default: 504)\n";
    os << "  --dump-raw PATH   Write the raw float32 depth map to PATH as a .npy file\n";
    os << "                    (multi-view: PATH is a directory or filename prefix)\n";
    os << "  --help            Show this help and exit\n";
    os << "\n";
    os << "Tasks, outputs and recommended checkpoints:\n";
    os << "  mono           output_depth.jpg                          DA3-SMALL/BASE/LARGE/GIANT\n";
    os << "  metric         output_depth.jpg + output_metric.npy     DA3METRIC-LARGE\n";
    os << "  multiview      output_depth_%03d.jpg (per frame)        DA3-LARGE/GIANT\n";
    os << "  pose           poses.txt + intrinsics.txt               DA3-LARGE/GIANT (not METRIC)\n";
    os << "  reconstruction output_cloud.ply                         DA3-GIANT; NESTED-GIANT-LARGE\n";
    os << "  gaussian       output_gaussians.ply (placeholder)       DA3-LARGE/GIANT + Gaussian head\n";
    os << "\n";
    os << "Camera presets (--camera) from the HF collection\n";
    os << "depth-anything/camera-depth-models (metric-oriented; prefer --task metric):\n";
    os << "  d405           Intel RealSense D405   camera-depth checkpoint\n";
    os << "  d435           Intel RealSense D435   camera-depth checkpoint\n";
    os << "  l515           Intel RealSense L515   camera-depth checkpoint\n";
    os << "  kinect         Microsoft Kinect        camera-depth checkpoint\n";
    os << "  zed2i-neural   Stereolabs ZED 2i       neural (stacked) variant\n";
    os << "  zed2i-quality  Stereolabs ZED 2i       quality (stacked) variant\n";
    os << "  base           Generic RGB camera      already metric by default;\n";
    os << "                                         skips scaling unless --focal given\n";
    os << "  No fx/fy values are baked in: pass --focal PX with your per-device\n";
    os << "  calibration. --focal always wins over the preset default.\n";
    os << "\n";
    os << "License: DA3-Small/Base/Mono/Metric are Apache-2.0; DA3-Large/Giant/Nested\n";
    os << "         are CC BY-NC 4.0 (non-commercial). Prefer the -1.1 weights.\n";
    os << "\n";
    os << "Note: Backend (ONNX Runtime or TensorRT) is selected at compile time.\n";
    os << "      Build with -DUSE_ONNX_RUNTIME=ON or -DUSE_TENSORRT=ON\n";
}

// ---------------------------------------------------------------------------
// Inference & output
// ---------------------------------------------------------------------------

// Run one image through the model and write whatever per-frame outputs are
// requested. Empty output paths are skipped. When @p frame_out is non-null the
// postprocessed depth, pose, intrinsics and (for PLY tasks) the original image
// are accumulated into it for the file-based tasks.
static void process_frame(DepthAnything3Inference &inference, const std::filesystem::path &image_path,
                          const Options &opts, const std::filesystem::path &depth_out,
                          const std::filesystem::path &raw_out, const std::filesystem::path &metric_out,
                          const std::filesystem::path &conf_out, FrameData *frame_out) {
    int orig_h = 0;
    int orig_w = 0;
    int out_h = 0;
    int out_w = 0;

    std::vector<float> input_data = inference.preprocess_image(image_path, orig_h, orig_w, out_h, out_w);
    std::cout << "Input image: " << image_path.string() << " (" << orig_h << "x" << orig_w << ")" << std::endl;
    std::cout << "Model input: " << out_h << "x" << out_w << std::endl;

    const auto t_infer_start = std::chrono::steady_clock::now();
    inference.run_inference(input_data, out_h, out_w);
    const auto t_infer_end = std::chrono::steady_clock::now();

    DepthResult result = inference.postprocess_outputs(out_h, out_w, orig_h, orig_w, opts.upscale_to_orig);

    if (opts.task == TaskType::metric) {
        apply_metric_scaling(result.depth, opts.focal, opts.skip_metric_scaling());
    }

    const double infer_ms = std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();
    std::cout << "Timing: infer " << infer_ms << " ms (H2D + GPU + D2H)" << std::endl;

    // Colorized depth map
    if (!depth_out.empty()) {
        const cv::Mat depth_vis = DepthAnything3Inference::colorize_depth(result.depth);
        if (const auto saved = DepthAnything3Inference::save_output_image(depth_vis, depth_out)) {
            std::cout << "Depth map saved to: " << saved->string() << std::endl;
        } else {
            throw std::runtime_error("Could not save depth map to " + depth_out.string());
        }
    }

    double min_depth = 0.0;
    double max_depth = 0.0;
    cv::minMaxLoc(result.depth, &min_depth, &max_depth);
    std::cout << "Depth range: [" << min_depth << ", " << max_depth << "]" << std::endl;

    // Metric .npy (float32 meters)
    if (!metric_out.empty()) {
        if (write_npy_float32(result.depth, metric_out)) {
            std::cout << "Metric depth map saved to: " << metric_out.string() << std::endl;
        } else {
            throw std::runtime_error("Could not save metric depth map to " + metric_out.string());
        }
    }

    // Raw .npy dump
    if (!raw_out.empty()) {
        if (write_npy_float32(result.depth, raw_out)) {
            std::cout << "Raw depth map saved to: " << raw_out.string() << std::endl;
        } else {
            throw std::runtime_error("Could not save raw depth map to " + raw_out.string());
        }
    }

    // Optional confidence map
    if (opts.output_conf && !conf_out.empty()) {
        if (!result.conf.empty()) {
            const cv::Mat conf_vis = DepthAnything3Inference::colorize_conf(result.conf);
            if (const auto saved = DepthAnything3Inference::save_output_image(conf_vis, conf_out)) {
                std::cout << "Confidence map saved to: " << saved->string() << std::endl;
            }
        } else {
            std::cout << "Note: Model did not produce a confidence output." << std::endl;
        }
    }

    if (frame_out != nullptr) {
        frame_out->depth = result.depth;
        const int h = result.depth.rows;
        const int w = result.depth.cols;
        // Default pinhole intrinsics: --focal when passed, else 1.2*W (R-5).
        const float focal = opts.focal_set ? opts.focal : tasks::default_focal(w);
        frame_out->intrinsics = {focal, focal, static_cast<float>(w) / 2.0f, static_cast<float>(h) / 2.0f};
        // Identity pose unless the loaded ONNX exposes a pose head.
        frame_out->pose_from_model = apply_model_pose(inference, frame_out->pose, frame_out->intrinsics);

        if (opts.task == TaskType::reconstruction || opts.task == TaskType::gaussian) {
            frame_out->image_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        }
    }
}

// Emit the accumulated file-based task outputs once every frame is processed.
static void write_task_outputs(const Options &opts, const std::vector<FrameData> &frames) {
    if (opts.task == TaskType::pose) {
        std::vector<std::array<float, 12>> poses;
        std::vector<std::array<float, 4>> intrinsics;
        poses.reserve(frames.size());
        intrinsics.reserve(frames.size());
        bool any_pose_head = false;
        for (const auto &frame : frames) {
            poses.push_back(frame.pose);
            intrinsics.push_back(frame.intrinsics);
            any_pose_head = any_pose_head || frame.pose_from_model;
        }
        if (!any_pose_head) {
            std::cerr << "Warning: loaded ONNX model exposes no pose head (extrinsics); wrote identity poses. "
                         "Export an any-view checkpoint with `export_onnx.py --task pose` for real poses."
                      << std::endl;
        }

        const std::filesystem::path poses_path = join_output(opts.output_dir, "poses.txt");
        const std::filesystem::path intrinsics_path = join_output(opts.output_dir, "intrinsics.txt");
        if (!tasks::write_poses_txt(poses_path, poses)) {
            throw std::runtime_error("Could not write poses to " + poses_path.string());
        }
        if (!tasks::write_intrinsics_txt(intrinsics_path, intrinsics)) {
            throw std::runtime_error("Could not write intrinsics to " + intrinsics_path.string());
        }
        std::cout << "Poses saved to: " << poses_path.string() << " (" << poses.size() << " frame(s))" << std::endl;
        std::cout << "Intrinsics saved to: " << intrinsics_path.string() << " (" << intrinsics.size() << " frame(s))"
                  << std::endl;
        if (opts.use_ray_pose) {
            std::cout << "Note: --use-ray-pose requested; ray-pose estimation is honored when the ONNX export "
                         "provides it."
                      << std::endl;
        }
    } else if (opts.task == TaskType::reconstruction) {
        const std::filesystem::path cloud_path = join_output(opts.output_dir, "output_cloud.ply");
        if (!tasks::save_point_cloud_ply(cloud_path, frames)) {
            throw std::runtime_error("Could not write point cloud to " + cloud_path.string());
        }
        std::cout << "Point cloud saved to: " << cloud_path.string() << std::endl;
    } else if (opts.task == TaskType::gaussian) {
        const std::filesystem::path gaussians_path = join_output(opts.output_dir, "output_gaussians.ply");
        if (!tasks::save_gaussians_ply(gaussians_path, frames)) {
            throw std::runtime_error("Could not write Gaussians to " + gaussians_path.string());
        }
        std::cout << "Gaussians saved to: " << gaussians_path.string() << std::endl;
        std::cout << "Note: placeholder Gaussians (opacity=1, scales from depth gradient, identity rotation, +Z "
                     "normal); the full Gaussian head (SH/scale/rotation) needs an upstream gsplat export."
                  << std::endl;
    }
}

int main(int argc, const char *argv[]) {
    const std::vector<std::string> args(argv + 1, argv + argc);

    // --help is handled first so it works without a model/image.
    for (const std::string &arg : args) {
        if (arg == "--help" || arg == "-h") {
            print_usage(std::cout, argv[0]);
            return 0;
        }
    }

    if (args.empty()) {
        print_usage(std::cerr, argv[0]);
        return 1;
    }

    Options opts;
    std::string task_name;
    std::vector<std::string> positionals;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string &arg = args[i];
        if (arg == "--task" && i + 1 < args.size()) {
            task_name = args[++i];
        } else if (arg == "--camera" && i + 1 < args.size()) {
            try {
                opts.camera = parse_camera(args[++i]);
            } catch (const std::exception &e) {
                std::cerr << "Invalid value for --camera: " << e.what() << std::endl;
                return 1;
            }
        } else if (arg == "--focal" && i + 1 < args.size()) {
            try {
                opts.focal = std::stof(args[++i]);
                if (opts.focal <= 0.0f) {
                    throw std::invalid_argument("focal must be positive");
                }
                opts.focal_set = true;
            } catch (const std::exception &e) {
                std::cerr << "Invalid value for --focal: " << e.what() << std::endl;
                return 1;
            }
        } else if (arg == "--input-dir" && i + 1 < args.size()) {
            opts.input_dir = args[++i];
        } else if (arg == "--output-dir" && i + 1 < args.size()) {
            opts.output_dir = args[++i];
        } else if (arg == "--already-metric") {
            opts.already_metric = true;
        } else if (arg == "--use-ray-pose") {
            opts.use_ray_pose = true;
        } else if (arg == "--conf") {
            opts.output_conf = true;
        } else if (arg == "--no-upscale") {
            opts.upscale_to_orig = false;
        } else if (arg == "--dump-raw" && i + 1 < args.size()) {
            opts.dump_raw = args[++i];
        } else if (arg == "--process-res" && i + 1 < args.size()) {
            try {
                opts.process_res = std::stoi(args[++i]);
                if (opts.process_res <= 0) {
                    throw std::invalid_argument("process_res must be positive");
                }
            } catch (const std::exception &e) {
                std::cerr << "Invalid value for --process-res: " << e.what() << std::endl;
                return 1;
            }
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Warning: ignoring unknown option '" << arg << "'" << std::endl;
        } else {
            positionals.push_back(arg);
        }
    }

    if (positionals.empty()) {
        std::cerr << "Error: missing <path_to_model>" << std::endl;
        print_usage(std::cerr, argv[0]);
        return 1;
    }

    const std::filesystem::path model_path = positionals[0];
    std::filesystem::path image_path;
    if (positionals.size() >= 2) {
        image_path = positionals[1];
    }

    // --input-dir overrides the positional image argument.
    std::filesystem::path input = image_path;
    if (!opts.input_dir.empty()) {
        input = opts.input_dir;
    }
    if (input.empty()) {
        std::cerr << "Error: missing <path_to_image> (or pass --input-dir DIR)" << std::endl;
        print_usage(std::cerr, argv[0]);
        return 1;
    }

    try {
        opts.task = parse_task(task_name);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    try {
        if (!std::filesystem::exists(input)) {
            throw std::runtime_error("Input path does not exist: " + input.string());
        }
        const bool input_is_dir = std::filesystem::is_directory(input);

        if (!opts.output_dir.empty()) {
            std::filesystem::create_directories(opts.output_dir);
        }

        std::cout << "Task: " << task_to_string(opts.task) << " (recommended: " << recommended_checkpoint(opts.task)
                  << ")" << std::endl;
        if (opts.use_ray_pose) {
            std::cout << "Note: --use-ray-pose requested." << std::endl;
        }
        if (opts.camera != CameraPreset::none) {
            std::cout << "Camera preset: " << camera_to_string(opts.camera) << " (" << camera_hf_id(opts.camera) << ")"
                      << std::endl;
            if (opts.task == TaskType::mono) {
                std::cout << "Note: camera-depth models are metric-oriented; prefer --task metric." << std::endl;
            }
            if (camera_implies_already_metric(opts.camera, opts.focal_set)) {
                std::cout << "Note: preset '" << camera_to_string(opts.camera)
                          << "' already outputs meters; skipping focal*net/300 scaling." << std::endl;
            }
        }

        Config config;
        config.process_res = opts.process_res;
        config.output_conf = opts.output_conf;

        const auto t_init_start = std::chrono::steady_clock::now();
        DepthAnything3Inference inference(model_path, config);
        const auto t_init_end = std::chrono::steady_clock::now();
        const double init_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();
        std::cout << "Timing: init " << init_ms << " ms (model load)" << std::endl;

        // File-based tasks accumulate every frame before writing a single artifact.
        const bool accumulate_frames = opts.task == TaskType::pose || opts.task == TaskType::reconstruction ||
                                       opts.task == TaskType::gaussian;
        std::vector<FrameData> frames;

        if (input_is_dir) {
            // Multi-view / directory mode: one indexed output per frame.
            if (opts.task == TaskType::multiview) {
                std::cerr << "Warning: multi-view per-frame depth is a baseline; spatially consistent fusion "
                             "requires an any-view ONNX export (R-7)."
                          << std::endl;
            }

            const std::vector<std::filesystem::path> inputs = collect_image_paths(input);
            if (inputs.empty()) {
                throw std::runtime_error("No images (*.jpg, *.jpeg, *.png) found in directory: " + input.string());
            }
            std::cout << "Found " << inputs.size() << " image(s) in " << input.string() << std::endl;

            for (std::size_t i = 0; i < inputs.size(); ++i) {
                const int index = static_cast<int>(i);
                const std::filesystem::path depth_out =
                    join_output(opts.output_dir, append_index("output_depth.jpg", index, ".jpg").string());

                std::filesystem::path raw_out;
                if (!opts.dump_raw.empty()) {
                    raw_out = raw_frame_path(opts.dump_raw, index);
                }

                std::filesystem::path metric_out;
                if (opts.task == TaskType::metric) {
                    metric_out = join_output(opts.output_dir, append_index("output_metric.npy", index, ".npy").string());
                }

                std::filesystem::path conf_out;
                if (opts.output_conf) {
                    conf_out = join_output(opts.output_dir, append_index("output_conf.jpg", index, ".jpg").string());
                }

                FrameData *frame_out = nullptr;
                if (accumulate_frames) {
                    frames.emplace_back();
                    frame_out = &frames.back();
                }
                process_frame(inference, inputs[i], opts, depth_out, raw_out, metric_out, conf_out, frame_out);
            }
        } else {
            // Single-image mode.
            std::filesystem::path metric_out;
            if (opts.task == TaskType::metric) {
                metric_out = join_output(opts.output_dir, "output_metric.npy");
            }

            FrameData *frame_out = nullptr;
            if (accumulate_frames) {
                frames.emplace_back();
                frame_out = &frames.back();
            }
            process_frame(inference, input, opts, join_output(opts.output_dir, "output_depth.jpg"), opts.dump_raw,
                          metric_out, join_output(opts.output_dir, "output_conf.jpg"), frame_out);
        }

        write_task_outputs(opts, frames);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
