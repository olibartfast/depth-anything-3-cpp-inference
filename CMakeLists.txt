cmake_minimum_required(VERSION 3.12)
project(depth_anything3_inference CXX)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Compiler Warnings (applied to project targets only, not third-party) ---
set(PROJECT_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
)

# Treat warnings as errors in CI (opt-in via -DWERROR=ON)
option(WERROR "Treat compiler warnings as errors" OFF)
if(WERROR)
    list(APPEND PROJECT_WARNING_FLAGS -Werror)
endif()

# Enable sanitizers in debug builds (opt-in via -DSANITIZERS=ON)
option(SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
endif()

# --- Backend Selection Options ---
option(USE_ONNX_RUNTIME "Build with ONNX Runtime backend support" ON)
option(USE_TENSORRT "Build with TensorRT backend support" OFF)

# Validate that at least one backend is enabled
if(NOT USE_ONNX_RUNTIME AND NOT USE_TENSORRT)
    message(FATAL_ERROR "At least one backend must be enabled. Set USE_ONNX_RUNTIME=ON or USE_TENSORRT=ON")
endif()

message(STATUS "Backend configuration:")
message(STATUS "  ONNX Runtime: ${USE_ONNX_RUNTIME}")
message(STATUS "  TensorRT:     ${USE_TENSORRT}")

# --- ONNX Runtime Configuration ---
if(USE_ONNX_RUNTIME)
    message(STATUS "Configuring ONNX Runtime backend...")

    set(ORT_VERSION "1.21.0")
    set(ORT_ARCHIVE_BASENAME "onnxruntime-linux-x64-${ORT_VERSION}")
    set(ORT_ARCHIVE_NAME "${ORT_ARCHIVE_BASENAME}.tgz")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE_NAME}")

    set(ORT_DEPS_DIR "${CMAKE_BINARY_DIR}/_deps")
    set(ORT_DOWNLOAD_PATH "${ORT_DEPS_DIR}/${ORT_ARCHIVE_NAME}")
    set(ORT_EXTRACT_DIR "${ORT_DEPS_DIR}")
    set(ORT_INSTALL_DIR "${ORT_EXTRACT_DIR}/${ORT_ARCHIVE_BASENAME}")

    if(NOT EXISTS "${ORT_INSTALL_DIR}/include/onnxruntime_cxx_api.h")
        message(STATUS "ONNX Runtime not found in build directory. Downloading and extracting...")
        file(MAKE_DIRECTORY "${ORT_DEPS_DIR}")
        if(NOT EXISTS "${ORT_DOWNLOAD_PATH}")
            message(STATUS "Downloading ONNX Runtime from ${ORT_URL}...")
            file(DOWNLOAD ${ORT_URL} ${ORT_DOWNLOAD_PATH}
                 SHOW_PROGRESS
                 TIMEOUT 600
                 STATUS download_status)
            list(GET download_status 0 download_code)
            list(GET download_status 1 download_message)
            if(NOT download_code EQUAL 0)
                message(FATAL_ERROR "Download failed: ${download_message}")
            endif()
            message(STATUS "Download completed: ${ORT_DOWNLOAD_PATH}")
        else()
            message(STATUS "Using existing ONNX Runtime archive: ${ORT_DOWNLOAD_PATH}")
        endif()
        message(STATUS "Extracting ${ORT_DOWNLOAD_PATH} to ${ORT_EXTRACT_DIR}...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${ORT_DOWNLOAD_PATH}"
            WORKING_DIRECTORY "${ORT_EXTRACT_DIR}"
            RESULT_VARIABLE extract_result
        )
        if(NOT extract_result EQUAL 0)
            message(FATAL_ERROR "Extraction failed with result: ${extract_result}")
        endif()
        message(STATUS "Extraction completed.")
        if(NOT EXISTS "${ORT_INSTALL_DIR}/include/onnxruntime_cxx_api.h")
            message(FATAL_ERROR "Extraction failed: expected directory not found at ${ORT_INSTALL_DIR}")
        endif()
    else()
        message(STATUS "Found existing ONNX Runtime at: ${ORT_INSTALL_DIR}")
    endif()

    set(ONNXRUNTIME_ROOTDIR "${ORT_INSTALL_DIR}" CACHE PATH "Path to ONNX Runtime root directory")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include")
    set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so.${ORT_VERSION}")

    if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime include directory not found at: ${ONNXRUNTIME_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
        message(FATAL_ERROR "ONNX Runtime library not found at: ${ONNXRUNTIME_LIBRARY}")
    endif()

    message(STATUS "ONNX Runtime include directory: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()

# --- TensorRT Configuration ---
# Pinned to the same version as rf-detr-cpp-inference (versions.env):
# TensorRT 11.2.1.2 built against CUDA 13.3. The 11.x Linux tarball is
# zstd-compressed and named
#   TensorRT-Enterprise-<version>-Linux-x86_64-cuda-<cuda>-Release-external.tar.zst
# and unpacks to TensorRT-<version>/. `cmake -E tar` detects the compression
# on extraction. Only TensorRT 11 is supported (strongly-typed network:
# engine precision follows the ONNX graph, no kFP16 flag).
if(USE_TENSORRT)
    message(STATUS "Configuring TensorRT backend...")

    set(TRT_VERSION "11.2.1.2" CACHE STRING "TensorRT version (four-component)")
    set(TRT_CUDA_VERSION "13.3" CACHE STRING "CUDA series TensorRT is built against")
    # Major.minor.patch truncation: used for the download URL directory.
    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" TRT_SHORT_VERSION "${TRT_VERSION}")

    if(TRT_VERSION VERSION_LESS 11)
        message(FATAL_ERROR "TRT_VERSION=${TRT_VERSION}: only TensorRT 11 is supported")
    endif()

    # NOTE: same Enterprise tarball layout as rf-detr-cpp-inference; it unpacks
    # to TensorRT-<version>/.
    set(TRT_ARCHIVE_NAME "TensorRT-Enterprise-${TRT_VERSION}-Linux-x86_64-cuda-${TRT_CUDA_VERSION}-Release-external.tar.zst")
    set(TRT_URL "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${TRT_SHORT_VERSION}/tars/${TRT_ARCHIVE_NAME}")

    # A pre-staged prefix (Colab/Brev gate scripts pre-download with wget
    # retries) wins over the in-build download.
    if(DEFINED ENV{TENSORRT_ROOTDIR} AND NOT TENSORRT_ROOTDIR)
        set(TENSORRT_ROOTDIR "$ENV{TENSORRT_ROOTDIR}" CACHE PATH "Path to TensorRT root directory")
    endif()

    set(TRT_DEPS_DIR "${CMAKE_BINARY_DIR}/_deps")
    set(TRT_DOWNLOAD_PATH "${TRT_DEPS_DIR}/${TRT_ARCHIVE_NAME}")
    set(TRT_EXTRACT_DIR "${TRT_DEPS_DIR}")
    set(TRT_INSTALL_DIR "${TRT_EXTRACT_DIR}/TensorRT-${TRT_VERSION}")

    if(TENSORRT_ROOTDIR AND EXISTS "${TENSORRT_ROOTDIR}/include/NvInfer.h")
        message(STATUS "Using pre-staged TensorRT at: ${TENSORRT_ROOTDIR}")
        set(TRT_INSTALL_DIR "${TENSORRT_ROOTDIR}")
    elseif(NOT EXISTS "${TRT_INSTALL_DIR}/include/NvInfer.h")
        message(STATUS "TensorRT not found in build directory. Downloading and extracting...")
        file(MAKE_DIRECTORY "${TRT_DEPS_DIR}")
        if(NOT EXISTS "${TRT_DOWNLOAD_PATH}")
            message(STATUS "Downloading TensorRT from ${TRT_URL}...")
            file(DOWNLOAD ${TRT_URL} ${TRT_DOWNLOAD_PATH}
                 SHOW_PROGRESS
                 TIMEOUT 600
                 STATUS download_status)
            list(GET download_status 0 download_code)
            list(GET download_status 1 download_message)
            if(NOT download_code EQUAL 0)
                message(FATAL_ERROR "Download failed: ${download_message}")
            endif()
            message(STATUS "Download completed: ${TRT_DOWNLOAD_PATH}")
        else()
            message(STATUS "Using existing TensorRT archive: ${TRT_DOWNLOAD_PATH}")
        endif()
        message(STATUS "Extracting ${TRT_DOWNLOAD_PATH} to ${TRT_EXTRACT_DIR} (format auto-detected)...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf "${TRT_DOWNLOAD_PATH}"
            WORKING_DIRECTORY "${TRT_EXTRACT_DIR}"
            RESULT_VARIABLE extract_result
        )
        if(NOT extract_result EQUAL 0)
            message(FATAL_ERROR "Extraction failed with result: ${extract_result}")
        endif()
        message(STATUS "Extraction completed.")
        if(NOT EXISTS "${TRT_INSTALL_DIR}/include/NvInfer.h")
            message(FATAL_ERROR "Extraction failed: expected directory not found at ${TRT_INSTALL_DIR}")
        endif()
    else()
        message(STATUS "Found existing TensorRT at: ${TRT_INSTALL_DIR}")
    endif()

    set(TENSORRT_ROOTDIR "${TRT_INSTALL_DIR}" CACHE PATH "Path to TensorRT root directory")
    set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOTDIR}/include")
    set(TENSORRT_LIBRARY "${TENSORRT_ROOTDIR}/lib/libnvinfer.so")
    set(TENSORRT_ONNX_PARSER "${TENSORRT_ROOTDIR}/lib/libnvonnxparser.so")

    # Find CUDA runtime (prefer the toolkit matching TRT_CUDA_VERSION)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CUDA_RUNTIME_LIBRARY CUDA::cudart)
    else()
        find_package(CUDA QUIET)
        if(NOT CUDA_FOUND)
            find_library(CUDA_RUNTIME_LIBRARY cudart
                HINTS
                    /usr/local/cuda-${TRT_CUDA_VERSION}/lib64
                    /usr/local/cuda/lib64
                    /usr/lib/x86_64-linux-gnu
                PATH_SUFFIXES lib lib64
            )
        else()
            set(CUDA_RUNTIME_LIBRARY ${CUDA_LIBRARIES})
        endif()
    endif()

    if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
        message(FATAL_ERROR "TensorRT include directory not found at: ${TENSORRT_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${TENSORRT_LIBRARY}")
        message(FATAL_ERROR "TensorRT library not found at: ${TENSORRT_LIBRARY}")
    endif()
    if(NOT EXISTS "${TENSORRT_ONNX_PARSER}")
        message(FATAL_ERROR "TensorRT ONNX parser library not found at: ${TENSORRT_ONNX_PARSER}")
    endif()
    if(NOT CUDA_RUNTIME_LIBRARY)
        message(FATAL_ERROR "CUDA runtime library not found. Please install CUDA toolkit.")
    endif()

    message(STATUS "TensorRT include directory: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
    message(STATUS "TensorRT ONNX parser: ${TENSORRT_ONNX_PARSER}")
    message(STATUS "CUDA runtime library: ${CUDA_RUNTIME_LIBRARY}")
endif()

# --- Find OpenCV ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)
find_package(Threads REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Install via: sudo apt-get install libopencv-dev")
endif()
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# --- Fetch Google Test ---
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# --- Source Directories ---
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(TEST_DIR "${CMAKE_SOURCE_DIR}/tests")

# --- Main Library ---
set(DA3_SOURCES
    "${SOURCE_DIR}/depth_anything3_inference.cpp"
    "${SOURCE_DIR}/processing_utils.cpp"
    "${SOURCE_DIR}/backends/inference_backend.cpp"
)

if(USE_ONNX_RUNTIME)
    list(APPEND DA3_SOURCES "${SOURCE_DIR}/backends/onnx_runtime_backend.cpp")
endif()

if(USE_TENSORRT)
    list(APPEND DA3_SOURCES "${SOURCE_DIR}/backends/tensorrt_backend.cpp")
endif()

add_library(da3_inference_lib STATIC ${DA3_SOURCES})
target_compile_options(da3_inference_lib PRIVATE ${PROJECT_WARNING_FLAGS})

target_include_directories(da3_inference_lib
    PUBLIC
        "${SOURCE_DIR}"
)

# Third-party includes as SYSTEM to suppress warnings from their headers
target_include_directories(da3_inference_lib SYSTEM
    PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
)

if(USE_ONNX_RUNTIME)
    target_include_directories(da3_inference_lib SYSTEM PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
    target_compile_definitions(da3_inference_lib PUBLIC USE_ONNX_RUNTIME)
endif()

if(USE_TENSORRT)
    target_include_directories(da3_inference_lib SYSTEM PUBLIC "${TENSORRT_INCLUDE_DIR}")
    target_compile_definitions(da3_inference_lib PUBLIC USE_TENSORRT)
endif()

target_link_libraries(da3_inference_lib
    PUBLIC
        "${OpenCV_LIBS}"
        Threads::Threads
)

if(USE_ONNX_RUNTIME)
    target_link_libraries(da3_inference_lib PUBLIC "${ONNXRUNTIME_LIBRARY}")
endif()

if(USE_TENSORRT)
    target_link_libraries(da3_inference_lib PUBLIC
        "${TENSORRT_LIBRARY}"
        "${TENSORRT_ONNX_PARSER}"
        "${CUDA_RUNTIME_LIBRARY}"
    )

    get_filename_component(TENSORRT_LIB_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
    set_target_properties(da3_inference_lib PROPERTIES
        BUILD_RPATH "${TENSORRT_LIB_DIR}"
        INSTALL_RPATH "${TENSORRT_LIB_DIR}"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# --- Main Application ---
add_executable(inference_app "${SOURCE_DIR}/main.cpp")
target_compile_options(inference_app PRIVATE ${PROJECT_WARNING_FLAGS})
target_link_libraries(inference_app PRIVATE da3_inference_lib)

if(USE_TENSORRT)
    get_filename_component(TENSORRT_LIB_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
    set_target_properties(inference_app PROPERTIES
        BUILD_RPATH "${TENSORRT_LIB_DIR}"
        INSTALL_RPATH "${TENSORRT_LIB_DIR}"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# Copy ONNX Runtime shared library next to the binary
if(USE_ONNX_RUNTIME)
    add_custom_command(TARGET inference_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIBRARY}"
                $<TARGET_FILE_DIR:inference_app>
        COMMENT "Copying ONNX Runtime library to output directory"
    )
endif()

# --- Unit Tests ---
set(TEST_FIXTURE_DIR "${TEST_DIR}/fixtures")
add_executable(unit_tests
    "${TEST_DIR}/unit/test_depth_anything3_inference.cpp"
    "${TEST_DIR}/unit/test_onnx_backend.cpp"
)
target_compile_options(unit_tests PRIVATE ${PROJECT_WARNING_FLAGS})
target_compile_definitions(unit_tests PRIVATE DA3_TEST_FIXTURE_DIR="${TEST_FIXTURE_DIR}")
target_link_libraries(unit_tests PRIVATE da3_inference_lib gtest gtest_main gmock)
target_include_directories(unit_tests PRIVATE "${TEST_DIR}/unit")

# --- Google Benchmark (opt-in via -DBENCHMARKS=ON) ---
option(BENCHMARKS "Build benchmark targets" OFF)
if(BENCHMARKS)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)

    add_executable(benchmarks "${TEST_DIR}/benchmark/bench_preprocessing.cpp")
    target_compile_options(benchmarks PRIVATE ${PROJECT_WARNING_FLAGS})
    target_link_libraries(benchmarks PRIVATE da3_inference_lib benchmark::benchmark)
endif()

# Enable testing
enable_testing()
add_test(NAME UnitTests COMMAND unit_tests)

# --- Custom Target to Run All Tests ---
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS unit_tests
    COMMENT "Running all tests..."
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
