#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "pellet/config.hpp"
#include "pellet/detector/frame_queue.hpp"

#if !defined(PELLET_CAMERA_WUST_VL_ONLY) || (PELLET_CAMERA_WUST_VL_ONLY != 1)
#error "Camera backend is locked to wust_vl + HikSDK. Define PELLET_CAMERA_WUST_VL_ONLY=1."
#endif

namespace wust_vl::video {
class Camera;
struct ImageFrame;
}  // namespace wust_vl::video

namespace pellet::detector {

class CaptureWorker {
 public:
  CaptureWorker(const CameraConfig& config, FrameQueue* frame_queue);
  ~CaptureWorker();

  bool Start();
  void Stop();

 private:
  void OnFrame(wust_vl::video::ImageFrame& frame);

  CameraConfig config_{};
  FrameQueue* frame_queue_{nullptr};
  std::atomic<bool> running_{false};
  std::atomic<bool> first_frame_received_{false};
  std::atomic<int64_t> startup_begin_ms_{0};
  std::atomic<int64_t> first_frame_latency_ms_{-1};
  std::string camera_sn_{"unknown"};
  int expected_width_{0};
  int expected_height_{0};
  std::unique_ptr<wust_vl::video::Camera> camera_;
  std::atomic<uint32_t> frame_id_{0};
};

}  // namespace pellet::detector
