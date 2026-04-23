#include "pellet/detector/capture_worker.hpp"

#if !defined(PELLET_CAMERA_WUST_VL_ONLY) || (PELLET_CAMERA_WUST_VL_ONLY != 1)
#error "Camera backend is locked to wust_vl + HikSDK. Define PELLET_CAMERA_WUST_VL_ONLY=1."
#endif

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <wust_vl/video/camera.hpp>

#include "pellet/utils/time_utils.hpp"

namespace pellet::detector {
namespace {

using wust_vl::video::ImageFrame;
using wust_vl::video::PixelFormat;

template <typename T>
T ReadYamlOr(const YAML::Node& node, const char* key, const T& default_value) {
  const YAML::Node child = node[key];
  if (!child) {
    return default_value;
  }
  try {
    return child.as<T>();
  } catch (const std::exception&) {
    return default_value;
  }
}

std::string ResolveWustVlConfigPath(const CameraConfig& config) {
  if (config.wust_vl_config_path.empty()) {
    return "config/camera.yaml";
  }
  return config.wust_vl_config_path;
}

int64_t ToSteadyMs(const std::chrono::steady_clock::time_point& tp) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
}

int64_t SteadyNowMs() {
  return ToSteadyMs(std::chrono::steady_clock::now());
}

void DebugLogIfEnabled(bool debug_mode, const std::string& message) {
  if (debug_mode) {
    std::cout << message << "\n";
  }
}

}  // namespace

CaptureWorker::CaptureWorker(const CameraConfig& config, FrameQueue* frame_queue)
    : config_(config), frame_queue_(frame_queue) {}

CaptureWorker::~CaptureWorker() {
  Stop();
}

bool CaptureWorker::Start() {
  if (running_.load() || frame_queue_ == nullptr) {
    return false;
  }

  YAML::Node camera_yaml;
  const std::string config_path = ResolveWustVlConfigPath(config_);
  try {
    camera_yaml = YAML::LoadFile(config_path);
  } catch (const std::exception& e) {
    std::cerr << "[capture_worker] failed to load wust_vl yaml: " << config_path
              << ", err: " << e.what() << "\n";
    return false;
  }

  const YAML::Node hik_camera = camera_yaml["hik_camera"];
  camera_sn_ = ReadYamlOr<std::string>(hik_camera, "target_sn", std::string("unknown"));
  expected_width_ = ReadYamlOr<int>(hik_camera, "width", 0);
  expected_height_ = ReadYamlOr<int>(hik_camera, "height", 0);
  DebugLogIfEnabled(
      config_.debug_mode,
      "[capture_worker] starting camera, sn=" + camera_sn_
          + ", expected_res=" + std::to_string(expected_width_) + "x"
          + std::to_string(expected_height_)
          + ", startup_timeout_ms=" + std::to_string(std::max(0, config_.startup_timeout_ms)));

  camera_ = std::make_unique<wust_vl::video::Camera>();
  if (!camera_->init(camera_yaml)) {
    std::cerr << "[capture_worker] wust_vl camera init failed\n";
    camera_.reset();
    return false;
  }

  first_frame_received_.store(false);
  first_frame_latency_ms_.store(-1);
  startup_begin_ms_.store(SteadyNowMs());
  running_.store(true);
  camera_->setFrameCallback([this](ImageFrame& frame) {
    OnFrame(frame);
  });
  camera_->start();

  const int timeout_ms = std::max(0, config_.startup_timeout_ms);
  if (timeout_ms > 0) {
    const auto begin = std::chrono::steady_clock::now();
    while (running_.load() && !first_frame_received_.load()) {
      const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - begin).count();
      if (elapsed_ms >= timeout_ms) {
        std::cerr << "[capture_worker] first frame startup timeout, sn=" << camera_sn_
                  << ", expected_res=" << expected_width_ << "x" << expected_height_
                  << ", timeout_ms=" << timeout_ms
                  << ", elapsed_ms=" << elapsed_ms << ", stop camera\n";
        Stop();
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  if (first_frame_received_.load()) {
    DebugLogIfEnabled(
        config_.debug_mode,
        "[capture_worker] startup ready, sn=" + camera_sn_
            + ", first_frame_latency_ms=" + std::to_string(first_frame_latency_ms_.load()));
  }
  return true;
}

void CaptureWorker::Stop() {
  running_.store(false);
  first_frame_received_.store(false);
  if (camera_) {
    camera_->stop();
    camera_->setFrameCallback({});
    camera_.reset();
  }
}

void CaptureWorker::OnFrame(ImageFrame& frame) {
  if (!running_.load() || frame_queue_ == nullptr || frame.src_img.empty()) {
    return;
  }

  // Pixel format convert path:
  cv::Mat frame_bgr;
#if PELLET_PIXELFMT_FIXED_BGR
  frame_bgr = frame.src_img;
#else
  switch (frame.pixel_format) {
    case PixelFormat::BGR:
      frame_bgr = frame.src_img;
      break;
    case PixelFormat::RGB:
      cv::cvtColor(frame.src_img, frame_bgr, cv::COLOR_RGB2BGR);
      break;
    case PixelFormat::GRAY:
      cv::cvtColor(frame.src_img, frame_bgr, cv::COLOR_GRAY2BGR);
      break;
    case PixelFormat::UNKNOWN:
    default:
      if (frame.src_img.channels() == 3) {
        frame_bgr = frame.src_img;
      } else if (frame.src_img.channels() == 1) {
        cv::cvtColor(frame.src_img, frame_bgr, cv::COLOR_GRAY2BGR);
      } else {
        return;
      }
      break;
  }
#endif

  FramePacket packet;
  packet.frame_id = frame_id_.fetch_add(1);
  packet.timestamp_ms = ToSteadyMs(frame.timestamp);
  if (packet.timestamp_ms <= 0) {
    packet.timestamp_ms = utils::NowMs();
  }
  packet.frame_bgr = std::move(frame_bgr);
  if (!first_frame_received_.exchange(true)) {
    const int64_t startup_begin_ms = startup_begin_ms_.load();
    int64_t latency_ms = -1;
    if (startup_begin_ms > 0) {
      latency_ms = std::max<int64_t>(0, SteadyNowMs() - startup_begin_ms);
    }
    first_frame_latency_ms_.store(latency_ms);
    DebugLogIfEnabled(
        config_.debug_mode,
        "[capture_worker] first frame received, sn=" + camera_sn_
            + ", actual_res=" + std::to_string(packet.frame_bgr.cols) + "x"
            + std::to_string(packet.frame_bgr.rows)
            + ", latency_ms=" + std::to_string(latency_ms));
  }
  frame_queue_->Push(packet);
}

}  // namespace pellet::detector
