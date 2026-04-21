#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>

#include "pellet/detector/detector_pipeline.h"

namespace pellet::detector {

class FrameQueue {
 public:
  explicit FrameQueue(std::size_t capacity);

  void Push(const FramePacket& frame);
  bool Pop(FramePacket* frame, int timeout_ms);
  void Stop();

 private:
  std::size_t capacity_{1};
  bool stopped_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<FramePacket> queue_;
};

}  // namespace pellet::detector
