#include "pellet/detector/detect_worker.hpp"

#include <chrono>

namespace pellet::detector {

DetectWorker::DetectWorker(
    FrameQueue* frame_queue,
    DetectorPipeline* pipeline,
    std::size_t max_buffered_results)
    : frame_queue_(frame_queue), pipeline_(pipeline), max_buffered_results_(max_buffered_results) {}

DetectWorker::~DetectWorker() {
  Stop();
}

void DetectWorker::Start() {
  if (running_.load() || frame_queue_ == nullptr || pipeline_ == nullptr) {
    return;
  }

  running_.store(true);
  worker_thread_ = std::thread(&DetectWorker::Run, this);
}

void DetectWorker::Stop() {
  running_.store(false);
  result_cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

bool DetectWorker::PopLatest(std::vector<Detection>* detections, int timeout_ms) {
  if (detections == nullptr) {
    return false;
  }

  std::unique_lock<std::mutex> lock(result_mutex_);
  result_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
    return !result_queue_.empty() || !running_.load();
  });

  if (result_queue_.empty()) {
    return false;
  }

  *detections = std::move(result_queue_.back());
  result_queue_.clear();
  return true;
}

void DetectWorker::Run() {
  while (running_.load()) {
    FramePacket frame;
    if (!frame_queue_->Pop(&frame, 50)) {
      continue;
    }

    std::vector<Detection> detections = pipeline_->Process(frame);

    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      if (result_queue_.size() >= max_buffered_results_) {
        result_queue_.pop_front();
      }
      result_queue_.push_back(std::move(detections));
    }

    result_cv_.notify_one();
  }
}

}  // namespace pellet::detector
