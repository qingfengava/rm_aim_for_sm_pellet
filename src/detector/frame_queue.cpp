#include "pellet/detector/frame_queue.hpp"

#include <algorithm>
#include <chrono>

namespace pellet::detector {

FrameQueue::FrameQueue(std::size_t capacity) : capacity_(std::max<std::size_t>(1, capacity)) {}

void FrameQueue::Push(const FramePacket& frame) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopped_) {
    return;
  }
  if (queue_.size() >= capacity_) {
    queue_.pop_front();
  }
  queue_.push_back(frame);
  cv_.notify_one();
}

bool FrameQueue::Pop(FramePacket* frame, int timeout_ms) {
  if (frame == nullptr) {
    return false;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
    return stopped_ || !queue_.empty();
  });

  if (queue_.empty()) {
    return false;
  }

  *frame = queue_.front();
  queue_.pop_front();
  return true;
}

void FrameQueue::Stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopped_ = true;
  cv_.notify_all();
}

}  // namespace pellet::detector
