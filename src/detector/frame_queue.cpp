#include "pellet/detector/frame_queue.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include <wust_vl/common/utils/logger.hpp>

#include "pellet/utils/debug_utils.hpp"

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

std::size_t FrameQueue::PruneStaleMetaLocked(const Clock::time_point& now) {
  if (meta_timestamps_.empty()) {
    return 0U;
  }
  const auto valid_window = std::chrono::milliseconds(queue_valid_ms_);
  std::size_t removed = 0U;
  while (!meta_timestamps_.empty() && (now - meta_timestamps_.front()) > valid_window) {
    meta_timestamps_.pop_front();
    ++removed;
  }
  if (removed > 0U) {
    dropped_stale_total_ += static_cast<std::uint64_t>(removed);
  }
  return removed;
}

void FrameQueue::LogDropIfNeededLocked() {
  if (dropped_overflow_total_ == 0U && dropped_stale_total_ == 0U) {
    return;
  }
  if (!utils::ShouldLogRateLimited("frame_queue", "frame_drop_detected")) {
    return;
  }
  WUST_WARN("frame_queue")
      << "frame drops observed, overflow_total=" << dropped_overflow_total_
      << ", stale_total=" << dropped_stale_total_
      << ", queue_size=" << meta_timestamps_.size()
      << ", capacity=" << capacity_
      << ", queue_valid_ms=" << queue_valid_ms_;
}

void FrameQueue::Push(const FramePacket& frame) {
  if (!timed_queue_.is_alive()) {
    return;
  }
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lock(meta_mutex_);
  (void)PruneStaleMetaLocked(now);

  //丢旧保新
  if (meta_timestamps_.size() >= capacity_) {
    FramePacket dropped;
    if (timed_queue_.pop_valid(dropped)) {
      if (!meta_timestamps_.empty()) {
        meta_timestamps_.pop_front();
        ++dropped_overflow_total_;
      }
    } else {
      dropped_overflow_total_ += static_cast<std::uint64_t>(meta_timestamps_.size());
      meta_timestamps_.clear();
    }
  }

  timed_queue_.push(frame, now);
  meta_timestamps_.push_back(now);
  ++pushed_total_;
  LogDropIfNeededLocked();
}

bool FrameQueue::Pop(FramePacket* frame, int timeout_ms) {
  if (frame == nullptr) {
    return false;
  }

  if (timeout_ms <= 0) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    const auto now = Clock::now();
    (void)PruneStaleMetaLocked(now);
    LogDropIfNeededLocked();
    if (!timed_queue_.pop_valid(*frame)) {
      return false;
    }
    if (!meta_timestamps_.empty()) {
      meta_timestamps_.pop_front();
    }
    ++popped_total_;
    return true;
  }

  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (Clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(meta_mutex_);
      const auto now = Clock::now();
      (void)PruneStaleMetaLocked(now);
      LogDropIfNeededLocked();
      if (timed_queue_.pop_valid(*frame)) {
        if (!meta_timestamps_.empty()) {
          meta_timestamps_.pop_front();
        }
        ++popped_total_;
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
  const auto now = Clock::now();
  (void)PruneStaleMetaLocked(now);
  LogDropIfNeededLocked();
  if (!timed_queue_.pop_valid(*frame)) {
    return false;
  }
  if (!meta_timestamps_.empty()) {
    meta_timestamps_.pop_front();
  }
  ++popped_total_;
  return true;
}

FrameQueue::StatsSnapshot FrameQueue::GetStatsSnapshot() {
  std::lock_guard<std::mutex> lock(meta_mutex_);
  StatsSnapshot snapshot;
  snapshot.drop_overflow = dropped_overflow_total_;
  snapshot.drop_stale = dropped_stale_total_;
  snapshot.push_total = pushed_total_;
  snapshot.pop_total = popped_total_;
  snapshot.queue_size = meta_timestamps_.size();
  snapshot.capacity = capacity_;
  return snapshot;
}

void FrameQueue::Stop() {
  timed_queue_.stop();
  std::lock_guard<std::mutex> lock(meta_mutex_);
  meta_timestamps_.clear();
}

}  // namespace pellet::detector
