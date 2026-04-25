#include "pellet/detector/frame_queue.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

namespace pellet::detector {
namespace {

int ClampMs(const int value, const int fallback) {
  return value > 0 ? value : fallback;
}

}  // namespace

FrameQueue::FrameQueue(std::size_t capacity, int queue_valid_ms, int pop_poll_ms)
    : capacity_(std::max<std::size_t>(1, capacity)),
      queue_valid_ms_(ClampMs(queue_valid_ms, 1000)),
      pop_poll_ms_(ClampMs(pop_poll_ms, 2)),
      timed_queue_(static_cast<double>(queue_valid_ms_) / 1000.0) {}

void FrameQueue::PruneStaleMetaLocked(const Clock::time_point& now) {
  if (meta_timestamps_.empty()) {
    return;
  }
  const auto valid_window = std::chrono::milliseconds(queue_valid_ms_);
  while (!meta_timestamps_.empty() && (now - meta_timestamps_.front()) > valid_window) {
    meta_timestamps_.pop_front();
  }
}

void FrameQueue::Push(const FramePacket& frame) {
  if (!timed_queue_.is_alive()) {
    return;
  }
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lock(meta_mutex_);
  PruneStaleMetaLocked(now);

  //丢旧保新
  if (meta_timestamps_.size() >= capacity_) {
    FramePacket dropped;
    if (timed_queue_.pop_valid(dropped)) {
      if (!meta_timestamps_.empty()) {
        meta_timestamps_.pop_front();
      }
    } else {
      meta_timestamps_.clear();
    }
  }

  timed_queue_.push(frame, now);
  meta_timestamps_.push_back(now);
}

bool FrameQueue::Pop(FramePacket* frame, int timeout_ms) {
  if (frame == nullptr) {
    return false;
  }

  if (timeout_ms <= 0) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    PruneStaleMetaLocked(Clock::now());
    if (!timed_queue_.pop_valid(*frame)) {
      return false;
    }
    if (!meta_timestamps_.empty()) {
      meta_timestamps_.pop_front();
    }
    return true;
  }

  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (Clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(meta_mutex_);
      PruneStaleMetaLocked(Clock::now());
      if (timed_queue_.pop_valid(*frame)) {
        if (!meta_timestamps_.empty()) {
          meta_timestamps_.pop_front();
        }
        return true;
      }
      if (!timed_queue_.is_alive()) {
        meta_timestamps_.clear();
        return false;
      }
    }

    const auto now = Clock::now();
    const auto remaining_ms =
        static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
                             .count());
    if (remaining_ms <= 0) {
      break;
    }
    const int sleep_ms = std::max(1, std::min(pop_poll_ms_, remaining_ms));
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  }

  std::lock_guard<std::mutex> lock(meta_mutex_);
  PruneStaleMetaLocked(Clock::now());
  if (!timed_queue_.pop_valid(*frame)) {
    return false;
  }
  if (!meta_timestamps_.empty()) {
    meta_timestamps_.pop_front();
  }
  return true;
}

void FrameQueue::Stop() {
  timed_queue_.stop();
  std::lock_guard<std::mutex> lock(meta_mutex_);
  meta_timestamps_.clear();
}

}  // namespace pellet::detector
