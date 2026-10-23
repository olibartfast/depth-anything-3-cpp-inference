# FindOpenCV.cmake — locate OpenCV without the monolithic libopencv-dev package.
#
# Ubuntu ships OpenCVConfig.cmake and opencv4.pc only in `libopencv-dev`, which
# also drags in libopencv-viz -> VTK -> OpenMPI -> AMD HIP (tens of megabytes
# that this project never uses). The per-component -dev packages carry the
# headers and linker symlinks but no CMake config, so this module resolves the
# requested components directly.
#
# Honours OpenCV_FIND_COMPONENTS (defaults below) and sets the classic
# variables: OpenCV_FOUND, OpenCV_INCLUDE_DIRS, OpenCV_LIBS, OpenCV_VERSION.

include(FindPackageHandleStandardArgs)

set(_opencv_default_components core imgproc imgcodecs highgui videoio)
if(NOT OpenCV_FIND_COMPONENTS)
    set(_opencv_components ${_opencv_default_components})
else()
    set(_opencv_components ${OpenCV_FIND_COMPONENTS})
endif()

find_path(OpenCV_INCLUDE_DIR
    NAMES opencv2/opencv.hpp
    PATH_SUFFIXES include/opencv4
)

set(_opencv_libs "")
set(_opencv_missing "")
foreach(_component IN LISTS _opencv_components)
    string(TOUPPER "${_component}" _component_upper)
    find_library(OpenCV_${_component_upper}_LIBRARY
        NAMES opencv_${_component}
    )
    if(OpenCV_${_component_upper}_LIBRARY)
        list(APPEND _opencv_libs "${OpenCV_${_component_upper}_LIBRARY}")
        set(OpenCV_${_component}_FOUND TRUE)
    else()
        set(OpenCV_${_component}_FOUND FALSE)
        list(APPEND _opencv_missing "${_component}")
    endif()
endforeach()

if(OpenCV_INCLUDE_DIR)
    file(STRINGS "${OpenCV_INCLUDE_DIR}/opencv2/core/version.hpp" _opencv_version_line
        REGEX "#define CV_VERSION_MAJOR")
    string(REGEX MATCH "[0-9]+" OpenCV_VERSION_MAJOR "${_opencv_version_line}")
    file(STRINGS "${OpenCV_INCLUDE_DIR}/opencv2/core/version.hpp" _opencv_version_minor
        REGEX "#define CV_VERSION_MINOR")
    string(REGEX MATCH "[0-9]+" OpenCV_VERSION_MINOR "${_opencv_version_minor}")
    file(STRINGS "${OpenCV_INCLUDE_DIR}/opencv2/core/version.hpp" _opencv_version_patch
        REGEX "#define CV_VERSION_REVISION")
    string(REGEX MATCH "[0-9]+" OpenCV_VERSION_PATCH "${_opencv_version_patch}")
    set(OpenCV_VERSION "${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_PATCH}")
endif()

find_package_handle_standard_args(OpenCV
    REQUIRED_VARS OpenCV_INCLUDE_DIR _opencv_libs
    VERSION_VAR OpenCV_VERSION
    HANDLE_COMPONENTS
)

if(OpenCV_FOUND)
    set(OpenCV_INCLUDE_DIRS "${OpenCV_INCLUDE_DIR}")
    set(OpenCV_LIBS "${_opencv_libs}")
endif()

mark_as_advanced(OpenCV_INCLUDE_DIR)
foreach(_component IN LISTS _opencv_components)
    string(TOUPPER "${_component}" _component_upper)
    mark_as_advanced(OpenCV_${_component_upper}_LIBRARY)
endforeach()
