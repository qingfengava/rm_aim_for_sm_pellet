#include "pellet/detector/capture_worker.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <wust_vl/common/utils/logger.hpp>
#include <yaml-cpp/yaml.h>
#include <wust_vl/video/camera.hpp>
#include <wust_vl/video/hik.hpp>

#include "pellet/detector/frame_queue.hpp"

namespace pellet::detector {
namespace {

using wust_vl::video::CameraType;
using wust_vl::video::ImageFrame;
using wust_vl::video::PixelFormat;

template <typename T> //安全读取 YAML
T ReadYamlOr(const YAML::Node& node, const char* key, const T& default_value) {
  const YAML::Node child = node[key];
  if (!child || child.IsNull() || !child.IsScalar()) {
    return default_value;
  }
  try {
    return child.as<T>();
  } catch (const YAML::BadConversion&) {
    return default_value;
  }
}

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

std::string CameraTypeToString(CameraType type) {
  switch (type) {
    case CameraType::HIK:
      return "HIK";
    case CameraType::VIDEO_PLAYER:
      return "VIDEO_PLAYER";
    case CameraType::UVC:
      return "UVC";
    default:
      return "UNKNOWN";
  }
}

int64_t ToSteadyMs(const std::chrono::steady_clock::time_point& tp) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
}

int64_t SteadyNowMs() {
  return ToSteadyMs(std::chrono::steady_clock::now());
}

void DebugLogIfEnabled(bool debug_mode, const std::string& message) {
  if (debug_mode) {
    WUST_INFO("capture_worker") << message;
  }
}

template <typename MessageBuilder>
void DebugLogLazyIfEnabled(bool debug_mode, MessageBuilder&& build_message) {
  static_assert(
      std::is_invocable_r_v<std::string, MessageBuilder>,
      "DebugLogLazyIfEnabled builder must be callable as std::string()");
  if (debug_mode) {
    WUST_INFO("capture_worker") << build_message();
  }
}

}  // namespace

CaptureWorker::CaptureWorker(
    const CameraConfig& config,
    FrameQueue* frame_queue,
    bool debug_mode)
    : config_(config), debug_mode_(debug_mode), frame_queue_(frame_queue) {}

CaptureWorker::~CaptureWorker() {
  Stop();
}

bool CaptureWorker::LoadCameraYaml(YAML::Node* camera_yaml) const {
  if (camera_yaml == nullptr) {
    return false;
  }

  const std::string config_path =
      config_.wust_vl_config_path.empty() ? "config/camera.yaml" : config_.wust_vl_config_path;
  try {
    *camera_yaml = YAML::LoadFile(config_path);
    return true;
  } catch (const std::exception& e) {
    WUST_ERROR("capture_worker") << "failed to load wust_vl yaml: " << config_path
                                 << ", err: " << e.what();
    return false;
  }
}

CaptureWorker::StartupConfig CaptureWorker::ParseStartupConfig(const YAML::Node& camera_yaml) {
  StartupConfig startup;
  const YAML::Node hik_camera = camera_yaml["hik_camera"];

  camera_sn_ = ReadYamlOr<std::string>(hik_camera, "target_sn", std::string("unknown"));
  expected_width_ = ReadYamlOr<int>(hik_camera, "width", 0);
  expected_height_ = ReadYamlOr<int>(hik_camera, "height", 0);

  startup.trigger_type =
      ReadYamlOr<std::string>(hik_camera, "trigger_type", std::string("none"));
  startup.runtime_control_enable =
      ReadYamlOr<bool>(hik_camera, "runtime_control_enable", false);
  startup.exposure_time = ReadYamlOr<double>(hik_camera, "exposure_time", -1.0);
  startup.gain = ReadYamlOr<double>(hik_camera, "gain", -1.0);
  startup.trigger_source = ReadYamlOr<std::string>(hik_camera, "trigger_source", std::string(""));
  startup.trigger_activation = ReadYamlOr<int64_t>(hik_camera, "trigger_activation", 0);
  software_trigger_interval_ms_ =
      std::max(0, ReadYamlOr<int>(hik_camera, "software_trigger_interval_ms", 0));
  startup.use_software_trigger = ToLower(startup.trigger_type) == "software";
  return startup;
}

void CaptureWorker::LogStartupSummary() const {
  DebugLogLazyIfEnabled(debug_mode_, [&]() {
    return "[capture_worker] starting camera, sn=" + camera_sn_
        + ", expected_res=" + std::to_string(expected_width_) + "x"
        + std::to_string(expected_height_)
        + ", startup_timeout_ms=" + std::to_string(std::max(0, config_.startup_timeout_ms));
  });
}

bool CaptureWorker::InitCameraDevice(const YAML::Node& camera_yaml) {
  camera_ = std::make_unique<wust_vl::video::Camera>();
  if (!camera_->init(camera_yaml)) {
    WUST_ERROR("capture_worker") << "wust_vl camera init failed";
    camera_.reset();
    hik_device_ = nullptr;
    return false;
  }

  hik_device_ = dynamic_cast<wust_vl::video::HikCamera*>(camera_->getDevice());
  if (hik_device_ == nullptr) {
    WUST_WARN("capture_worker")
        << "wust_vl device is not HikCamera, runtime controls(exposure/gain/trigger) are "
           "disabled";
  } else {
    DebugLogIfEnabled(debug_mode_, "[capture_worker] HikCamera device attached");
  }
  return true;
}

void CaptureWorker::LogDeviceSummary() const {
  std::string camera_type_name = "UNKNOWN";
  if (camera_) {
    camera_type_name = CameraTypeToString(camera_->type_);
  }
  DebugLogLazyIfEnabled(debug_mode_, [&]() {
    return "[capture_worker] device_type=" + camera_type_name
        + ", hik_downcast_ok=" + std::string(hik_device_ != nullptr ? "true" : "false");
  });
}

void CaptureWorker::ApplyRuntimeControls(const StartupConfig& startup) {
  if (!startup.runtime_control_enable || hik_device_ == nullptr) {
    return;
  }

  if (!startup.trigger_type.empty()) {
    const bool trigger_ok =
        SetTrigger(startup.trigger_type, startup.trigger_source, startup.trigger_activation);
    if (!trigger_ok) {
      WUST_WARN("capture_worker") << "failed to apply startup trigger config";
    }
  }
  if (startup.exposure_time > 0.0) {
    const bool exposure_ok = SetExposureTime(startup.exposure_time);
    if (!exposure_ok) {
      WUST_WARN("capture_worker") << "failed to apply startup exposure config";
    }
  }
  if (startup.gain >= 0.0) {
    const bool gain_ok = SetGain(startup.gain);
    if (!gain_ok) {
      WUST_WARN("capture_worker") << "failed to apply startup gain config";
    }
  }

  DebugLogLazyIfEnabled(debug_mode_, [&]() {
    return "[capture_worker] runtime controls applied, trigger_type="
        + (startup.trigger_type.empty() ? std::string("keep_yaml") : startup.trigger_type)
        + ", exposure=" + std::to_string(startup.exposure_time)
        + ", gain=" + std::to_string(startup.gain);
  });
}

void CaptureWorker::LogRuntimeSummary(const StartupConfig& startup) const {
  const double current_exposure = hik_device_ != nullptr ? hik_device_->getExposureTime() : -1.0;
  const double current_gain = hik_device_ != nullptr ? hik_device_->getGain() : -1.0;
  DebugLogLazyIfEnabled(debug_mode_, [&]() {
    return "[capture_worker] trigger_type=" + startup.trigger_type
        + ", trigger_source=" + startup.trigger_source
        + ", trigger_activation=" + std::to_string(startup.trigger_activation)
        + ", software_trigger_interval_ms=" + std::to_string(software_trigger_interval_ms_)
        + ", exposure=" + std::to_string(current_exposure)
        + ", gain=" + std::to_string(current_gain);
  });
}

void CaptureWorker::StartCapturePath(const StartupConfig& startup) {
  software_trigger_mode_.store(startup.use_software_trigger && hik_device_ != nullptr);
  DebugLogLazyIfEnabled(debug_mode_, [this]() {
    return "[capture_worker] frame_path="
        + std::string(software_trigger_mode_.load() ? "callback+active_readImage"
                                                    : "callback_only");
  });

  running_.store(true);
  camera_->setFrameCallback([this](ImageFrame& frame) {
    OnFrame(frame);
  });
  camera_->start();

  if (software_trigger_mode_.load()) {
    DebugLogIfEnabled(
        debug_mode_,
        "[capture_worker] software trigger mode enabled, start active readImage loop");
    software_trigger_thread_ = std::thread(&CaptureWorker::RunSoftwareTriggerLoop, this);
  }
}

bool CaptureWorker::WaitFirstFrameOrTimeout(int timeout_ms) {
  if (timeout_ms <= 0) {
    return true;
  }

  const auto begin = std::chrono::steady_clock::now();
  while (running_.load() && !first_frame_received_.load()) {
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - begin).count();
    if (elapsed_ms >= timeout_ms) {
      WUST_ERROR("capture_worker")
          << "first frame startup timeout, sn=" << camera_sn_
          << ", expected_res=" << expected_width_ << "x" << expected_height_
          << ", timeout_ms=" << timeout_ms
          << ", elapsed_ms=" << elapsed_ms;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return true;
}

bool CaptureWorker::Start() {
  if (running_.load() || frame_queue_ == nullptr) {
    return false;
  }

  YAML::Node camera_yaml;
  if (!LoadCameraYaml(&camera_yaml)) {
    return false;
  }

  const StartupConfig startup = ParseStartupConfig(camera_yaml);
  LogStartupSummary();

  if (!InitCameraDevice(camera_yaml)) {
    return false;
  }
  LogDeviceSummary();

  first_frame_received_.store(false);
  first_frame_latency_ms_.store(-1);
  startup_begin_ms_.store(SteadyNowMs());

  ApplyRuntimeControls(startup);
  LogRuntimeSummary(startup);
  StartCapturePath(startup);

  if (!WaitFirstFrameOrTimeout(std::max(0, config_.startup_timeout_ms))) {
    Stop();
    return false;
  }

  if (first_frame_received_.load()) {
    DebugLogLazyIfEnabled(debug_mode_, [this]() {
      return "[capture_worker] startup ready, sn=" + camera_sn_
          + ", first_frame_latency_ms=" + std::to_string(first_frame_latency_ms_.load());
    });
  }

  return true;
}

void CaptureWorker::Stop() {
  //running=false -> stop active loop -> camera.stop -> clear callback -> clear hik handle
  running_.store(false);

  software_trigger_mode_.store(false);
  first_frame_received_.store(false);
  if (software_trigger_thread_.joinable()) {
    software_trigger_thread_.join();
  }

  if (camera_) {
    camera_->stop();
    camera_->setFrameCallback({});
  }
  
  hik_device_ = nullptr;
  camera_.reset();
}

bool CaptureWorker::SetExposureTime(double exposure_time) {
  if (hik_device_ == nullptr) {
    WUST_WARN("capture_worker") << "SetExposureTime ignored, HikCamera unavailable";
    return false;
  }
  hik_device_->setExposureTime(exposure_time);
  return true;
}

bool CaptureWorker::SetGain(double gain) {
  if (hik_device_ == nullptr) {
    WUST_WARN("capture_worker") << "SetGain ignored, HikCamera unavailable";
    return false;
  }
  hik_device_->setGain(gain);
  return true;
}

bool CaptureWorker::SetTrigger(
    const std::string& trigger_type,
    const std::string& trigger_source,
    int64_t trigger_activation) {
  if (hik_device_ == nullptr) {
    WUST_WARN("capture_worker") << "SetTrigger ignored, HikCamera unavailable";
    return false;
  }
  const auto type_enum = hik_device_->string2TriggerType(trigger_type);
  const bool ok = hik_device_->setTrigger(type_enum, trigger_source, trigger_activation);
  if (!ok) {
    WUST_WARN("capture_worker") << "failed to set trigger, type=" << trigger_type
                                << ", source=" << trigger_source
                                << ", activation=" << trigger_activation;
  }
  return ok;
}

void CaptureWorker::RunSoftwareTriggerLoop() {
  const auto min_trigger_interval = std::chrono::milliseconds(software_trigger_interval_ms_);
  auto last_trigger_time = std::chrono::steady_clock::time_point{};

  while (running_.load() && software_trigger_mode_.load()) {
    if (min_trigger_interval.count() > 0) {
      const auto now = std::chrono::steady_clock::now();
      if (last_trigger_time.time_since_epoch().count() != 0) {
        const auto next_allowed = last_trigger_time + min_trigger_interval;
        if (now < next_allowed) {
          std::this_thread::sleep_for(next_allowed - now);
        }
      }
      last_trigger_time = std::chrono::steady_clock::now();
    }

    if (!camera_) {
      break;
    }
    ImageFrame frame = camera_->readImage();
    if (!running_.load()) {
      break;
    }
    if (frame.src_img.empty()) {
      continue;
    }
    OnFrame(frame);
  }
}

void CaptureWorker::OnFrame(ImageFrame& frame) {
  if (!running_.load() || frame_queue_ == nullptr || frame.src_img.empty()) {
    return;
  }

  //统一图像格式
  cv::Mat frame_bgr;
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

  FramePacket packet;
  packet.frame_id = frame_id_.fetch_add(1, std::memory_order_relaxed);
  packet.timestamp_ms = ToSteadyMs(frame.timestamp);
  if (packet.timestamp_ms <= 0) {
    packet.timestamp_ms = SteadyNowMs();
  }
  packet.frame_bgr = std::move(frame_bgr);
  if (!first_frame_received_.exchange(true)) {
    const int64_t startup_begin_ms = startup_begin_ms_.load();
    int64_t latency_ms = -1;
    if (startup_begin_ms > 0) {
      latency_ms = std::max<int64_t>(0, SteadyNowMs() - startup_begin_ms);
    }
    first_frame_latency_ms_.store(latency_ms);
    DebugLogLazyIfEnabled(debug_mode_, [&]() {
      return "[capture_worker] first frame received, sn=" + camera_sn_
          + ", actual_res=" + std::to_string(packet.frame_bgr.cols) + "x"
          + std::to_string(packet.frame_bgr.rows)
          + ", latency_ms=" + std::to_string(latency_ms)
          + ", frame_path="
          + std::string(software_trigger_mode_.load() ? "callback+active_readImage"
                                                      : "callback_only");
    });
  }
  frame_queue_->Push(packet);
}

}  // namespace pellet::detector
