#include "pellet/detector/detect_worker.hpp"

#include <algorithm>
#include <chrono>

#include "pellet/detector/detector_pipeline.hpp"
#include "pellet/detector/frame_queue.hpp"

namespace pellet::detector {

DetectWorker::DetectWorker(
    FrameQueue* frame_queue,
    DetectorPipeline* pipeline,
    int frame_pop_timeout_ms,
    bool thread_monitor_enable,
    bool show_thread_status)
    : frame_queue_(frame_queue),
      pipeline_(pipeline),
      frame_pop_timeout_ms_(std::max(1, frame_pop_timeout_ms)),
      thread_monitor_enable_(thread_monitor_enable),
      show_thread_status_(show_thread_status) {}

DetectWorker::~DetectWorker() {
  Stop();
}

void DetectWorker::Start() {
  if (running_.load(std::memory_order_relaxed) || frame_queue_ == nullptr || pipeline_ == nullptr) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(result_mutex_);
    latest_result_.clear();
    has_latest_result_ = false;
  }
  running_.store(true, std::memory_order_relaxed);
  worker_thread_ = wust_vl::common::concurrency::MonitoredThread::create(
      kWorkerThreadName,
      [this](const wust_vl::common::concurrency::MonitoredThread::Ptr& thread_handle) {
        Run(thread_handle);
      });
  if (thread_monitor_enable_ && worker_thread_) {
    wust_vl::common::concurrency::ThreadManager::instance().registerThread(worker_thread_);
    if (show_thread_status_) {
      wust_vl::common::concurrency::ThreadManager::instance().printStatus();
    }
  }
}

void DetectWorker::Stop() {
  running_.store(false, std::memory_order_relaxed);
  result_cv_.notify_all();
  if (worker_thread_) {
    if (thread_monitor_enable_ && show_thread_status_) {
      wust_vl::common::concurrency::ThreadManager::instance().printStatus();
    }
    worker_thread_->stop();
    if (thread_monitor_enable_) {
      wust_vl::common::concurrency::ThreadManager::instance().unregisterThread(
          kWorkerThreadName);
      if (show_thread_status_) {
        wust_vl::common::concurrency::ThreadManager::instance().printStatus();
      }
    }
    worker_thread_.reset();
  }
}

bool DetectWorker::PopLatest(std::vector<Detection>* detections, int timeout_ms) {
  if (detections == nullptr) {
    return false;
  }

  std::unique_lock<std::mutex> lock(result_mutex_);
  result_cv_.wait_for(lock, std::chrono::milliseconds(std::max(0, timeout_ms)), [this] {
    return has_latest_result_ || !running_.load(std::memory_order_relaxed);
  });

  if (!has_latest_result_) {
    return false;
  }

  *detections = std::move(latest_result_);
  latest_result_.clear();//输出后立即清空
  has_latest_result_ = false;
  return true;
}

void DetectWorker::Run(
    const wust_vl::common::concurrency::MonitoredThread::Ptr& thread_handle) {
  constexpr auto kHeartbeatInterval = std::chrono::milliseconds(200);
  auto next_heartbeat = std::chrono::steady_clock::now();

  while (running_.load(std::memory_order_relaxed)) {
    if (thread_handle) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= next_heartbeat) {
        thread_handle->heartbeat();
        next_heartbeat = now + kHeartbeatInterval;
      }
    }

    FramePacket frame;
    if (!frame_queue_->Pop(&frame, frame_pop_timeout_ms_)) {
      continue;
    }

    std::vector<Detection> detections = pipeline_->Process(frame);//逐帧处理
    if (detections.empty()) {
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      latest_result_ = std::move(detections);
      has_latest_result_ = true;
    }

    result_cv_.notify_one();
  }
}

}  // namespace pellet::detector
