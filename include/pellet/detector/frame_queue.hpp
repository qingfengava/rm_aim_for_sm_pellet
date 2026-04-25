#pragma once

#include <cstddef>
#include <chrono>
#include <deque>
#include <mutex>

#include <wust_vl/common/concurrency/queues.hpp>

#include "pellet/detector/frame_packet.hpp"

namespace pellet::detector {

class FrameQueue {
 public:
  FrameQueue(std::size_t capacity, int queue_valid_ms = 1000, int pop_poll_ms = 2);

  void Push(const FramePacket& frame);
  bool Pop(FramePacket* frame, int timeout_ms);
  void Stop();

 private:
  using Clock = std::chrono::steady_clock;

  void PruneStaleMetaLocked(const Clock::time_point& now);

  std::size_t capacity_{1};
  int queue_valid_ms_{1000};
  int pop_poll_ms_{2};
  wust_vl::common::concurrency::TimedQueue<FramePacket> timed_queue_;
  std::mutex meta_mutex_;
  std::deque<Clock::time_point> meta_timestamps_;
};

}  // namespace pellet::detector
