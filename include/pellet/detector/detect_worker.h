#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "pellet/detector/detector_pipeline.h"
#include "pellet/detector/frame_queue.h"
#include "pellet/type.h"

namespace pellet::detector {

class DetectWorker {
 public:
  DetectWorker(FrameQueue* frame_queue, DetectorPipeline* pipeline, std::size_t max_buffered_results = 8);
  ~DetectWorker();

  void Start();
  void Stop();
  bool PopLatest(std::vector<Detection>* detections, int timeout_ms);

 private:
  void Run();

  FrameQueue* frame_queue_{nullptr};
  DetectorPipeline* pipeline_{nullptr};
  std::size_t max_buffered_results_{8};

  std::atomic<bool> running_{false};
  std::thread worker_thread_;

  std::mutex result_mutex_;
  std::condition_variable result_cv_;
  std::deque<std::vector<Detection>> result_queue_;
};

}  // namespace pellet::detector
