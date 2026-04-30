#pragma once

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>

#include <wust_vl/common/concurrency/queues.hpp>

#include "pellet/detector/frame_packet.hpp"

namespace pellet::detector {

class FrameQueue {
 public:
  struct StatsSnapshot {
    std::uint64_t drop_overflow{0};
    std::uint64_t drop_stale{0};
    std::uint64_t push_total{0};
    std::uint64_t pop_total{0};
    std::size_t queue_size{0};
    std::size_t capacity{0};
  };

  FrameQueue(std::size_t capacity, int queue_valid_ms = 1000, int pop_poll_ms = 2);

  void Push(const FramePacket& frame);
  bool Pop(FramePacket* frame, int timeout_ms);
  void Stop();
  StatsSnapshot GetStatsSnapshot();

 private:
  using Clock = std::chrono::steady_clock;

  std::size_t PruneStaleMetaLocked(const Clock::time_point& now);
  void LogDropIfNeededLocked();

  std::size_t capacity_{1};
  int queue_valid_ms_{1000};
  int pop_poll_ms_{2};
  wust_vl::common::concurrency::TimedQueue<FramePacket> timed_queue_;
  std::mutex meta_mutex_;
  std::deque<Clock::time_point> meta_timestamps_;
  std::uint64_t dropped_overflow_total_{0};
  std::uint64_t dropped_stale_total_{0};
  std::uint64_t pushed_total_{0};
  std::uint64_t popped_total_{0};
};

}  // namespace pellet::detector
