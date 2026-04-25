#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>
#include <wust_vl/common/concurrency/monitored_thread.hpp>

#include "pellet/type.hpp"

namespace pellet::detector {

class DetectorPipeline;
class FrameQueue;

class DetectWorker {
 public:
  DetectWorker(
      FrameQueue* frame_queue,
      DetectorPipeline* pipeline,
      int frame_pop_timeout_ms = 100,
      bool thread_monitor_enable = true,
      bool show_thread_status = false);
  ~DetectWorker();

  void Start();
  void Stop();
  bool PopLatest(std::vector<Detection>* detections, int timeout_ms);

 private:
  void Run(const wust_vl::common::concurrency::MonitoredThread::Ptr& thread_handle);

  FrameQueue* frame_queue_{nullptr};
  DetectorPipeline* pipeline_{nullptr};
  int frame_pop_timeout_ms_{100};

  static constexpr const char* kWorkerThreadName = "detector_worker";
  std::atomic<bool> running_{false};
  bool thread_monitor_enable_{true};
  bool show_thread_status_{false};
  wust_vl::common::concurrency::MonitoredThread::Ptr worker_thread_;

  std::mutex result_mutex_;
  std::condition_variable result_cv_;
  std::vector<Detection> latest_result_;
  bool has_latest_result_{false};
};

}  // namespace pellet::detector
