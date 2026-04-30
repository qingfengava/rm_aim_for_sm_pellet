cmake_minimum_required(VERSION 3.16)
project(rm_aim_for_sm_pellets LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#====================================
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
option(ENABLE_TENSORRT "Enable TensorRT backend" ON)
option(ENABLE_OPENVINO "Enable OpenVINO backend" OFF)
option(ENABLE_NCNN "Enable ncnn backend" OFF)
#====================================
option(BUILD_TOOLS "Build tools targets" ON)
option(PELLET_ENABLE_LTO "Enable LTO(IPO) for Release and RelWithDebInfo builds" ON)
#====================================

if(NOT DEFINED BUILD_TESTING)
    set(BUILD_TESTING OFF CACHE BOOL "Build tests")
endif()

if(PELLET_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT PELLET_IPO_SUPPORTED OUTPUT PELLET_IPO_OUTPUT)
    if(PELLET_IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
        message(STATUS "IPO/LTO enabled for Release and RelWithDebInfo.")
    else()
        message(WARNING "IPO/LTO requested but not supported: ${PELLET_IPO_OUTPUT}")
    endif()
endif()

# Fallback to local wust_vl CMake modules for FindHikSDK.cmake / FindTensorRT.cmake
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../wust_vl/cmake")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../wust_vl/cmake")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc video)
find_package(HikSDK REQUIRED)
find_package(wust_vl CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)

set(PELLET_WUST_VL_TARGET "")
if(TARGET wust_vl::wust_vl)
    set(PELLET_WUST_VL_TARGET wust_vl::wust_vl)
elseif(TARGET wust_vl::wust_vl_video)
    set(PELLET_WUST_VL_TARGET wust_vl::wust_vl_video)
else()
    message(FATAL_ERROR "wust_vl package found, but neither wust_vl::wust_vl nor wust_vl::wust_vl_video is available.")
endif()

set(PELLET_YAML_TARGET "")
if(TARGET yaml-cpp::yaml-cpp)
    set(PELLET_YAML_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(PELLET_YAML_TARGET yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp package found, but no known imported target is available.")
endif()

set(PELLET_CORE_SOURCES
    src/config.cpp
    src/detector/detector.cpp
    src/detector/detector_pipeline.cpp
    src/detector/capture_worker.cpp
    src/detector/detect_worker.cpp
    src/detector/frame_queue.cpp
    src/imgprocess/preprocess.cpp
    src/imgprocess/bg_subtractor.cpp
    src/imgprocess/motion_confirm.cpp
    src/imgprocess/morphology.cpp
    src/imgprocess/candidate_extractor.cpp
    src/imgprocess/candidate_filter.cpp
    src/imgprocess/candidate_nms.cpp
    src/imgprocess/roi_cropper.cpp
    src/infer/onnx_classifier.cpp
    src/infer/mock_classifier.cpp
    src/infer/tensorrt_classifier.cpp
    src/infer/openvino_classifier.cpp
    src/infer/ncnn_classifier.cpp
    src/utils/debug_utils.cpp
    src/utils/infer_utils.cpp
)

add_library(pellet_core ${PELLET_CORE_SOURCES})
target_include_directories(pellet_core PUBLIC include)
target_link_libraries(
    pellet_core
    PUBLIC
        ${OpenCV_LIBS}
        ${PELLET_WUST_VL_TARGET}
        ${PELLET_YAML_TARGET}
)

# Keep mock backend only for test builds.
if(BUILD_TESTING)
    target_compile_definitions(pellet_core PRIVATE PELLET_ENABLE_MOCK_BACKEND=1)
endif()

if(ENABLE_ONNXRUNTIME)
    find_path(
        ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS
            ${ONNXRUNTIME_ROOT}
            $ENV{ONNXRUNTIME_ROOT}
        PATH_SUFFIXES include include/onnxruntime/core/session onnxruntime/core/session
    )
    find_library(
        ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS
            ${ONNXRUNTIME_ROOT}
            $ENV{ONNXRUNTIME_ROOT}
        PATH_SUFFIXES lib lib64
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        target_compile_definitions(pellet_core PRIVATE PELLET_WITH_ORT=1)
        target_include_directories(pellet_core PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(pellet_core PRIVATE ${ONNXRUNTIME_LIBRARY})
        message(STATUS "ONNX Runtime enabled.")
    else()
        message(STATUS "ONNX Runtime not found. Build continues with mock backend available.")
    endif()
endif()

if(ENABLE_TENSORRT)
    find_package(CUDAToolkit REQUIRED)
    find_package(TensorRT REQUIRED)
    if(NOT TARGET TensorRT::TensorRT)
        message(FATAL_ERROR "TensorRT package found but target TensorRT::TensorRT is missing.")
    endif()
    foreach(PELLET_CUDA_TARGET CUDA::cudart CUDA::nppc CUDA::nppicc)
        if(NOT TARGET ${PELLET_CUDA_TARGET})
            message(FATAL_ERROR "CUDAToolkit package is missing required target: ${PELLET_CUDA_TARGET}")
        endif()
    endforeach()
    if(TARGET TensorRT::TensorRT AND TARGET CUDA::cudart)
        target_compile_definitions(pellet_core PRIVATE PELLET_WITH_TENSORRT=1)
        target_link_libraries(pellet_core PRIVATE TensorRT::TensorRT CUDA::cudart)
        message(STATUS "TensorRT enabled.")
    else()
        message(STATUS "TensorRT targets unavailable. Build continues with mock backend available.")
    endif()
endif()

if(ENABLE_OPENVINO)
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(TARGET openvino::runtime)
        target_compile_definitions(pellet_core PRIVATE PELLET_WITH_OPENVINO=1)
        target_link_libraries(pellet_core PRIVATE openvino::runtime)
        message(STATUS "OpenVINO enabled.")
    else()
        find_library(
            OPENVINO_RUNTIME_LIBRARY
            NAMES openvino libopenvino.so.2600
            HINTS
                /opt/intel/openvino_2026.0.0/runtime
                /opt/intel/openvino/runtime
            PATH_SUFFIXES
                lib
                lib64
                lib/intel64
                runtime/lib
                runtime/lib/intel64
        )
        find_path(
            OPENVINO_INCLUDE_DIR
            NAMES openvino/openvino.hpp
            HINTS
                /opt/intel/openvino_2026.0.0/runtime
                /opt/intel/openvino/runtime
            PATH_SUFFIXES
                include
                runtime/include
        )
        if(OPENVINO_RUNTIME_LIBRARY AND OPENVINO_INCLUDE_DIR)
            target_compile_definitions(pellet_core PRIVATE PELLET_WITH_OPENVINO=1)
            target_include_directories(pellet_core PRIVATE ${OPENVINO_INCLUDE_DIR})
            target_link_libraries(pellet_core PRIVATE ${OPENVINO_RUNTIME_LIBRARY})
            get_filename_component(OPENVINO_RUNTIME_DIR "${OPENVINO_RUNTIME_LIBRARY}" DIRECTORY)
            set_property(TARGET pellet_core APPEND PROPERTY BUILD_RPATH "${OPENVINO_RUNTIME_DIR}")
            message(STATUS "OpenVINO enabled via fallback path: ${OPENVINO_RUNTIME_LIBRARY}")
        else()
            message(STATUS "OpenVINO not found. Build continues with mock backend available.")
        endif()
    endif()
endif()

if(ENABLE_NCNN)
    find_package(ncnn QUIET CONFIG)
    if(TARGET ncnn)
        target_compile_definitions(pellet_core PRIVATE PELLET_WITH_NCNN=1)
        target_link_libraries(pellet_core PRIVATE ncnn)
        message(STATUS "ncnn enabled.")
    else()
        find_path(
            NCNN_INCLUDE_DIR
            NAMES ncnn/net.h
            HINTS
                ${NCNN_ROOT}
                $ENV{NCNN_ROOT}
                /usr/local
                /usr
            PATH_SUFFIXES include
        )
        find_library(
            NCNN_LIBRARY
            NAMES ncnn
            HINTS
                ${NCNN_ROOT}
                $ENV{NCNN_ROOT}
                /usr/local
                /usr
            PATH_SUFFIXES lib lib64
        )
        if(NCNN_INCLUDE_DIR AND NCNN_LIBRARY AND NOT NCNN_LIBRARY MATCHES "\\.a$")
            target_compile_definitions(pellet_core PRIVATE PELLET_WITH_NCNN=1)
            target_include_directories(pellet_core PRIVATE ${NCNN_INCLUDE_DIR})
            target_link_libraries(pellet_core PRIVATE ${NCNN_LIBRARY})
            message(STATUS "ncnn enabled.")
        elseif(NCNN_INCLUDE_DIR AND NCNN_LIBRARY)
            message(STATUS "ncnn static library detected without CMake package. Skipping auto-enable to avoid missing transitive links.")
        else()
            message(STATUS "ncnn not found. Build continues with mock backend available.")
        endif()
    endif()
endif()

add_executable(pellet_detector src/main.cpp)
target_link_libraries(pellet_detector PRIVATE pellet_core)

if(BUILD_TOOLS)
    add_executable(benchmark_detector tools/benchmark_detector.cpp)
    target_link_libraries(benchmark_detector PRIVATE pellet_core)
    add_executable(recorder tools/recorder.cpp)
    target_link_libraries(recorder PRIVATE pellet_core)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        function(pellet_add_gtest test_name test_source)
            add_executable(${test_name} ${test_source})
            target_link_libraries(${test_name} PRIVATE pellet_core GTest::gtest_main)
            add_test(NAME ${test_name} COMMAND ${test_name})
        endfunction()

        pellet_add_gtest(test_bg_subtractor tests/unit/test_bg_subtractor.cpp)
        pellet_add_gtest(test_debug_utils tests/unit/test_debug_utils.cpp)
        pellet_add_gtest(test_candidate_filter tests/unit/test_candidate_filter.cpp)
        pellet_add_gtest(test_candidate_extractor tests/unit/test_candidate_extractor.cpp)
        pellet_add_gtest(test_candidate_nms tests/unit/test_candidate_nms.cpp)
        pellet_add_gtest(test_roi_cropper tests/unit/test_roi_cropper.cpp)
        pellet_add_gtest(test_morphology tests/unit/test_morphology.cpp)
        pellet_add_gtest(test_frame_queue_wust_vl tests/unit/test_frame_queue_wust_vl.cpp)
        pellet_add_gtest(test_detect_worker_threading tests/unit/test_detect_worker_threading.cpp)
        pellet_add_gtest(test_detector_e2e tests/integration/test_detector_e2e.cpp)
    else()
        message(WARNING "BUILD_TESTING is ON but GTest was not found. Skipping test targets.")
    endif()
endif()
