#ifdef USE_TENSORRT

#include "tensorrt_backend.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace da3::backend {

void TensorRTBackend::Logger::log(Severity severity, const char *msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

TensorRTBackend::TensorRTBackend() {
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
}

TensorRTBackend::~TensorRTBackend() {
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

#if NV_TENSORRT_MAJOR >= 10
    const int num_bindings = engine_->getNbIOTensors();
#else
    const int num_bindings = engine_->getNbBindings();
#endif
    std::cout << "[TensorRT] Model has " << num_bindings << " bindings" << std::endl;

    std::vector<int64_t> detected_shape = input_shape;

    for (int i = 0; i < num_bindings; ++i) {
#if NV_TENSORRT_MAJOR >= 10
        const char *name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        bool is_input = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
#else
        const char *name = engine_->getBindingName(i);
        auto dims = engine_->getBindingDimensions(i);
        bool is_input = engine_->bindingIsInput(i);
#endif

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
        binding_size *= sizeof(float);

        void *device_buffer = nullptr;
        cudaMalloc(&device_buffer, binding_size);
        device_buffers_.push_back(device_buffer);
    }

    for (const auto &shape : output_shapes_) {
        size_t size = 1;
        for (auto dim : shape) {
            size *= static_cast<size_t>(dim);
        }
        host_output_buffers_.emplace_back(size);
    }

    std::cout << "[TensorRT] Initialization complete" << std::endl;
    return detected_shape;
}

bool TensorRTBackend::build_engine_from_onnx(const std::filesystem::path &model_path,
                                              const std::vector<int64_t> & /*input_shape*/) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder, TensorRTDeleter>(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return false;
    }

    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network =
        std::unique_ptr<nvinfer1::INetworkDefinition, TensorRTDeleter>(builder->createNetworkV2(explicit_batch));
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

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig, TensorRTDeleter>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return false;
    }

#if NV_TENSORRT_MAJOR >= 10 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
#else
    config->setMaxWorkspaceSize(1ULL << 30);
#endif

    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TensorRT] FP16 mode enabled" << std::endl;
    }

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

    file.write(static_cast<const char *>(serialized->data()), serialized->size());
}

bool TensorRTBackend::deserialize_engine(const std::filesystem::path &engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), static_cast<std::streamsize>(size))) {
        std::cerr << "Failed to read engine file" << std::endl;
        return false;
    }

#if NV_TENSORRT_MAJOR >= 10
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
#else
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size, nullptr));
#endif
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    std::cout << "[TensorRT] Engine loaded successfully" << std::endl;
    return true;
}

void TensorRTBackend::run_inference(std::span<const float> input_data,
                                    const std::vector<int64_t> & /*input_shape*/) {
    const size_t input_size = input_data.size() * sizeof(float);
    cudaMemcpy(device_buffers_[static_cast<size_t>(input_binding_index_)], input_data.data(), input_size,
               cudaMemcpyHostToDevice);

#if NV_TENSORRT_MAJOR >= 10
    for (size_t i = 0; i < device_buffers_.size(); ++i) {
        const char *name = engine_->getIOTensorName(static_cast<int>(i));
        context_->setTensorAddress(name, device_buffers_[i]);
    }
    if (!context_->enqueueV3(nullptr)) {
        throw std::runtime_error("TensorRT inference execution failed");
    }
    cudaStreamSynchronize(nullptr);
#else
    if (!context_->executeV2(device_buffers_.data())) {
        throw std::runtime_error("TensorRT inference execution failed");
    }
#endif

    for (size_t i = 0; i < output_binding_indices_.size(); ++i) {
        const int binding_idx = output_binding_indices_[i];
        const size_t output_size = host_output_buffers_[i].size() * sizeof(float);
        cudaMemcpy(host_output_buffers_[i].data(), device_buffers_[static_cast<size_t>(binding_idx)], output_size,
                   cudaMemcpyDeviceToHost);
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
