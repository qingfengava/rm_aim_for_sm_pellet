#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "pellet/config.hpp"

namespace wust_vl::video {
class Camera;
class HikCamera;
struct ImageFrame;
}  // namespace wust_vl::video

namespace YAML {
class Node;
}  // namespace YAML

namespace pellet::detector {

class FrameQueue;

class CaptureWorker {
 public:
  CaptureWorker(const CameraConfig& config, FrameQueue* frame_queue, bool debug_mode = false);
  ~CaptureWorker();

  bool Start();
  void Stop();
  bool SetExposureTime(double exposure_time);
  bool SetGain(double gain);
  bool SetTrigger(
      const std::string& trigger_type,
      const std::string& trigger_source,
      int64_t trigger_activation);

 private:
  struct StartupConfig {
    bool runtime_control_enable{false};
    std::string trigger_type{"none"};
    std::string trigger_source{};
    int64_t trigger_activation{0};
    bool use_software_trigger{false};
    double exposure_time{-1.0};
    double gain{-1.0};
  };

  bool LoadCameraYaml(YAML::Node* camera_yaml) const;
  StartupConfig ParseStartupConfig(const YAML::Node& camera_yaml);
  void LogStartupSummary() const;
  bool InitCameraDevice(const YAML::Node& camera_yaml);
  void LogDeviceSummary() const;
  void ApplyRuntimeControls(const StartupConfig& startup);
  void LogRuntimeSummary(const StartupConfig& startup) const;
  void StartCapturePath(const StartupConfig& startup);
  bool WaitFirstFrameOrTimeout(int timeout_ms);

  void OnFrame(wust_vl::video::ImageFrame& frame);
  void RunSoftwareTriggerLoop();

  CameraConfig config_{};
  bool debug_mode_{false};
  FrameQueue* frame_queue_{nullptr};
  std::atomic<bool> running_{false};
  std::atomic<bool> first_frame_received_{false};
  std::atomic<int64_t> startup_begin_ms_{0};
  std::atomic<int64_t> first_frame_latency_ms_{-1};
  std::string camera_sn_{"unknown"};
  int expected_width_{0};
  int expected_height_{0};
  std::unique_ptr<wust_vl::video::Camera> camera_;
  wust_vl::video::HikCamera* hik_device_{nullptr};
  std::atomic<bool> software_trigger_mode_{false};
  int software_trigger_interval_ms_{0};
  std::thread software_trigger_thread_;
  std::atomic<uint32_t> frame_id_{0};
  std::mutex unsupported_pixel_mutex_;
  uint64_t unsupported_pixel_drop_total_{0};
};

}  // namespace pellet::detector
