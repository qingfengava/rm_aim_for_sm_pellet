#pragma once

#include <atomic>
#include <thread>

#include <opencv2/videoio.hpp>

#include "pellet/config.h"
#include "pellet/detector/frame_queue.h"

namespace pellet::detector {

class CaptureWorker {
 public:
  CaptureWorker(const CameraConfig& config, FrameQueue* frame_queue);
  ~CaptureWorker();

  bool Start();
  void Stop();

 private:
  void Run();

  CameraConfig config_{};
  FrameQueue* frame_queue_{nullptr};
  std::atomic<bool> running_{false};
  std::thread worker_thread_;
  cv::VideoCapture capture_;
  uint32_t frame_id_{0};
};

}  // namespace pellet::detector
