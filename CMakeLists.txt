cmake_minimum_required(VERSION 3.16)
project(rm_aim_for_sm_pellets LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#====================================
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
option(ENABLE_TENSORRT "Enable TensorRT backend" ON)
option(ENABLE_OPENVINO "Enable OpenVINO backend" ON)
option(ENABLE_NCNN "Enable ncnn backend" ON)
#====================================

option(BUILD_TOOLS "Build tools targets" ON)
option(BUILD_TESTING "Build tests" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)

set(PELLET_CORE_SOURCES
    src/config.cpp
    src/detector/detector.cpp
    src/detector/detector_pipeline.cpp
    src/detector/capture_worker.cpp
    src/detector/detect_worker.cpp
    src/detector/frame_queue.cpp
    src/imgprocess/preprocess.cpp
    src/imgprocess/three_frame_diff.cpp
    src/imgprocess/binarizer.cpp
    src/imgprocess/morphology.cpp
    src/imgprocess/candidate_extractor.cpp
    src/imgprocess/candidate_filter.cpp
    src/imgprocess/roi_cropper.cpp
    src/infer/onnx_classifier.cpp
    src/infer/mock_classifier.cpp
    src/infer/tensorrt_classifier.cpp
    src/infer/openvino_classifier.cpp
    src/infer/ncnn_classifier.cpp
    src/utils/time_utils.cpp
    src/utils/image_utils.cpp
)

add_library(pellet_core ${PELLET_CORE_SOURCES})
target_include_directories(pellet_core PUBLIC include)
target_link_libraries(pellet_core PUBLIC ${OpenCV_LIBS})

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
    find_path(
        TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h
        HINTS
            ${TENSORRT_ROOT}
            $ENV{TENSORRT_ROOT}
            /usr/include
            /usr/include/x86_64-linux-gnu
        PATH_SUFFIXES include
    )
    find_library(
        TENSORRT_NVINFER
        NAMES nvinfer
        HINTS
            ${TENSORRT_ROOT}
            $ENV{TENSORRT_ROOT}
            /usr/lib
            /usr/lib/x86_64-linux-gnu
        PATH_SUFFIXES lib lib64
    )
    find_path(
        CUDA_RUNTIME_INCLUDE_DIR
        NAMES cuda_runtime_api.h
        HINTS
            ${CUDA_ROOT}
            $ENV{CUDA_ROOT}
            /usr/local/cuda
            /usr
        PATH_SUFFIXES include
    )
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER AND CUDA_RUNTIME_INCLUDE_DIR)
        target_compile_definitions(pellet_core PRIVATE PELLET_WITH_TENSORRT=1)
        target_include_directories(pellet_core PRIVATE ${TENSORRT_INCLUDE_DIR} ${CUDA_RUNTIME_INCLUDE_DIR})
        target_link_libraries(pellet_core PRIVATE ${TENSORRT_NVINFER})
        message(STATUS "TensorRT enabled.")
    else()
        message(STATUS "TensorRT prerequisites not found (NvInfer, nvinfer, cuda_runtime_api). Build continues with mock backend available.")
    endif()
endif()

if(ENABLE_OPENVINO)
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(TARGET openvino::runtime)
        target_compile_definitions(pellet_core PRIVATE PELLET_WITH_OPENVINO=1)
        target_link_libraries(pellet_core PRIVATE openvino::runtime)
        message(STATUS "OpenVINO enabled.")
    else()
        message(STATUS "OpenVINO not found. Build continues with mock backend available.")
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
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_executable(test_three_frame_diff tests/unit/test_three_frame_diff.cpp)
        target_link_libraries(test_three_frame_diff PRIVATE pellet_core GTest::gtest_main)
        add_test(NAME test_three_frame_diff COMMAND test_three_frame_diff)

        add_executable(test_candidate_filter tests/unit/test_candidate_filter.cpp)
        target_link_libraries(test_candidate_filter PRIVATE pellet_core GTest::gtest_main)
        add_test(NAME test_candidate_filter COMMAND test_candidate_filter)

        add_executable(test_roi_cropper tests/unit/test_roi_cropper.cpp)
        target_link_libraries(test_roi_cropper PRIVATE pellet_core GTest::gtest_main)
        add_test(NAME test_roi_cropper COMMAND test_roi_cropper)

        add_executable(test_morphology tests/unit/test_morphology.cpp)
        target_link_libraries(test_morphology PRIVATE pellet_core GTest::gtest_main)
        add_test(NAME test_morphology COMMAND test_morphology)

        add_executable(test_detector_e2e tests/integration/test_detector_e2e.cpp)
        target_link_libraries(test_detector_e2e PRIVATE pellet_core GTest::gtest_main)
        add_test(NAME test_detector_e2e COMMAND test_detector_e2e)
    else()
        message(WARNING "BUILD_TESTING is ON but GTest was not found. Skipping test targets.")
    endif()
endif()
