#ifdef USE_TENSORRT

#include "tensorrt_backend.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace da3::backend {

void TensorRTBackend::Logger::log(Severity severity, const char *msg) noexcept {
    // Filter out INFO messages for cleaner output
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

TensorRTBackend::TensorRTBackend() {
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream for TensorRT backend");
    }
}

TensorRTBackend::~TensorRTBackend() {
    if (stream_ != nullptr) {
        // Outstanding async work references device_buffers_; drain before freeing.
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    // Free CUDA device buffers
    for (void *buffer : device_buffers_) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
}

std::vector<int64_t> TensorRTBackend::initialize(const std::filesystem::path &model_path,
                                                 const std::vector<int64_t> &input_shape) {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path.string());
    }

    bool engine_loaded = false;
    std::filesystem::path engine_path;

    std::string extension = model_path.extension().string();
    if (extension == ".engine" || extension == ".trt") {
        std::cout << "[TensorRT] Loading pre-built engine: " << model_path << std::endl;
        engine_loaded = deserialize_engine(model_path);
        if (!engine_loaded) {
            throw std::runtime_error("Failed to load TensorRT engine from: " + model_path.string());
        }
    } else {
        engine_path = model_path;
        engine_path.replace_extension(".engine");

        if (std::filesystem::exists(engine_path)) {
            std::cout << "[TensorRT] Found cached engine: " << engine_path << std::endl;
            engine_loaded = deserialize_engine(engine_path);
        }

        if (!engine_loaded) {
            std::cout << "[TensorRT] Building engine from ONNX model..." << std::endl;
            if (!build_engine_from_onnx(model_path, input_shape)) {
                throw std::runtime_error("Failed to build TensorRT engine from ONNX");
            }
            serialize_engine(engine_path);
            std::cout << "[TensorRT] Engine saved to: " << engine_path << std::endl;
        }
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    const int num_bindings = engine_->getNbIOTensors();
    std::cout << "[TensorRT] Model has " << num_bindings << " bindings" << std::endl;

    std::vector<int64_t> detected_shape = input_shape;

    for (int i = 0; i < num_bindings; ++i) {
        const char *name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        auto io_mode = engine_->getTensorIOMode(name);
        bool is_input = (io_mode == nvinfer1::TensorIOMode::kINPUT);
        // Every buffer below is sized and copied as float32. A reduced-precision ONNX
        // must keep its graph I/O in float32, or the copies would silently read and
        // write the wrong number of bytes.
        if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error(std::string("TensorRT binding '") + name +
                                     "' is not float32; convert the model to FP16 with its I/O kept in "
                                     "float32");
        }

        std::cout << "  Binding " << i << ": " << name << (is_input ? " (input)" : " (output)") << " - Shape: [";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        if (is_input) {
            input_binding_index_ = i;
            if (dims.nbDims >= 3) {
                detected_shape.clear();
                for (int j = 0; j < dims.nbDims; ++j) {
                    detected_shape.push_back(dims.d[j]);
                }
            }
        } else {
            output_binding_indices_.push_back(i);
            std::vector<int64_t> shape;
            for (int j = 0; j < dims.nbDims; ++j) {
                shape.push_back(dims.d[j]);
            }
            output_shapes_.push_back(shape);
        }

        size_t binding_size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            binding_size *= static_cast<size_t>(dims.d[j]);
        }
        if (is_input) {
            input_element_count_ = binding_size;
        }
        binding_size *= sizeof(float);

        void *device_buffer = nullptr;
        if (cudaMalloc(&device_buffer, binding_size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA device buffer");
        }
        device_buffers_.push_back(device_buffer);
    }

    for (const auto &shape : output_shapes_) {
        const size_t size = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                            [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
        host_output_buffers_.emplace_back(size);
    }

    std::cout << "[TensorRT] Initialization complete" << std::endl;
    return detected_shape;
}

bool TensorRTBackend::build_engine_from_onnx(const std::filesystem::path &model_path,
                                             const std::vector<int64_t> & /*input_shape*/) {
    // The ONNX graph carries its own static input shape; no optimization profile is set here.
    auto builder = std::unique_ptr<nvinfer1::IBuilder, TensorRTDeleter>(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return false;
    }

    // Explicit batch is the only mode, so createNetworkV2 takes no flag.
    const uint32_t network_flags = 0U;
    auto network =
        std::unique_ptr<nvinfer1::INetworkDefinition, TensorRTDeleter>(builder->createNetworkV2(network_flags));
    if (!network) {
        std::cerr << "Failed to create TensorRT network" << std::endl;
        return false;
    }

    auto parser =
        std::unique_ptr<nvonnxparser::IParser, TensorRTDeleter>(nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return false;
    }

    if (!parser->parseFromFile(model_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "  Error " << i << ": " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    // This backend sets no optimization profile, so the ONNX graph must carry
    // static shapes. export_onnx.py writes dynamic spatial axes by default;
    // re-export with --fixed-size for the TensorRT path.
    for (int i = 0; i < network->getNbInputs(); ++i) {
        const auto *input = network->getInput(i);
        const auto dims = input->getDimensions();
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] < 0) {
                std::cerr << "ONNX input '" << input->getName() << "' has dynamic dim " << j
                          << ". TensorRT needs static shapes: re-export with "
                          << "`python export_onnx.py --fixed-size --process-res <N>`." << std::endl;
                return false;
            }
        }
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig, TensorRTDeleter>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return false;
    }

    // Workspace limit (1GB)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // TensorRT 11 removed weak typing together with BuilderFlag::kFP16: every network is
    // strongly typed, so each layer runs in the precision the ONNX graph declares. An FP32
    // export builds an FP32 engine; FP16 comes from converting the ONNX beforehand.
    std::cout << "[TensorRT] Strongly typed build: engine precision follows the ONNX model "
                 "(convert it to FP16 beforehand for an FP16 engine)"
              << std::endl;

    std::cout << "[TensorRT] Building engine... This may take a few minutes." << std::endl;
    engine_.reset(builder->buildEngineWithConfig(*network, *config));
    if (!engine_) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return false;
    }

    std::cout << "[TensorRT] Engine built successfully" << std::endl;
    return true;
}

void TensorRTBackend::serialize_engine(const std::filesystem::path &engine_path) {
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory, TensorRTDeleter>(engine_->serialize());
    if (!serialized) {
        throw std::runtime_error("Failed to serialize TensorRT engine");
    }

    std::ofstream file(engine_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + engine_path.string());
    }

    file.write(static_cast<const char *>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
}

bool TensorRTBackend::deserialize_engine(const std::filesystem::path &engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    const std::streamoff end = file.tellg();
    if (end < 0) {
        std::cerr << "Failed to determine engine file size: " << engine_path << std::endl;
        return false;
    }
    const size_t size = static_cast<size_t>(end);
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), static_cast<std::streamsize>(size))) {
        std::cerr << "Failed to read engine file" << std::endl;
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    std::cout << "[TensorRT] Engine loaded successfully" << std::endl;
    return true;
}

void TensorRTBackend::enqueue() {
    // enqueueV3 takes tensor addresses set via setTensorAddress
    for (int i = 0; i < static_cast<int>(device_buffers_.size()); ++i) {
        const char *name = engine_->getIOTensorName(i);
        context_->setTensorAddress(name, device_buffers_[static_cast<size_t>(i)]);
    }
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT inference execution failed");
    }
}

void TensorRTBackend::run_inference(std::span<const float> input_data, const std::vector<int64_t> &input_shape) {
    // Static engine: the preprocessed input must exactly fill the input binding.
    // (TRT input is square fixed-size; use a square image at the export resolution.)
    if (input_data.size() != input_element_count_) {
        std::string shape_str = "[";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            shape_str += std::to_string(input_shape[i]);
            if (i + 1 < input_shape.size()) {
                shape_str += ", ";
            }
        }
        shape_str += "]";
        throw std::runtime_error("TensorRT input size mismatch: engine expects " +
                                 std::to_string(input_element_count_) + " floats, got " +
                                 std::to_string(input_data.size()) + " (shape " + shape_str +
                                 "). The engine is fixed-shape — rerun with the export resolution.");
    }
    // Host path: async copies and enqueue on stream_, then one synchronisation.
    // The host output buffers are fully populated when this returns, but nothing
    // is pushed onto the legacy default stream any more.
    const size_t input_size = input_data.size() * sizeof(float);
    if (cudaMemcpyAsync(device_buffers_[static_cast<size_t>(input_binding_index_)], input_data.data(), input_size,
                        cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        throw std::runtime_error("Failed to copy input tensor to device");
    }

    enqueue();

    for (size_t i = 0; i < output_binding_indices_.size(); ++i) {
        const int binding_idx = output_binding_indices_[i];
        const size_t output_size = host_output_buffers_[i].size() * sizeof(float);
        if (cudaMemcpyAsync(host_output_buffers_[i].data(), device_buffers_[static_cast<size_t>(binding_idx)],
                            output_size, cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
            throw std::runtime_error("Failed to copy output tensor from device");
        }
    }

    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        throw std::runtime_error("TensorRT stream synchronization failed");
    }
}

size_t TensorRTBackend::get_output_count() const { return output_binding_indices_.size(); }

void TensorRTBackend::get_output_data(size_t output_index, float *data, size_t size) {
    if (output_index >= host_output_buffers_.size()) {
        throw std::out_of_range("Output index out of range");
    }

    const auto &buffer = host_output_buffers_[output_index];
    if (buffer.size() != size) {
        throw std::runtime_error("Output tensor size mismatch. Expected: " + std::to_string(size) +
                                 ", Got: " + std::to_string(buffer.size()));
    }

    std::copy(buffer.begin(), buffer.end(), data);
}

std::vector<int64_t> TensorRTBackend::get_output_shape(size_t output_index) const {
    if (output_index >= output_shapes_.size()) {
        throw std::out_of_range("Output index out of range");
    }

    return output_shapes_[output_index];
}

} // namespace da3::backend

#endif // USE_TENSORRT
