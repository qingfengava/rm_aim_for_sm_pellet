#include "pellet/detector/detect_worker.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>

#include <wust_vl/common/utils/logger.hpp>

#include "pellet/detector/detector_pipeline.hpp"
#include "pellet/detector/frame_queue.hpp"

namespace pellet::detector {

DetectWorker::DetectWorker(
    FrameQueue* frame_queue,
    DetectorPipeline* pipeline,
    int frame_pop_timeout_ms,
    bool thread_monitor_enable,
    bool show_thread_status,
    bool enable_pipeline_stats,
    bool show_stats_1s)
    : frame_queue_(frame_queue),
      pipeline_(pipeline),
      frame_pop_timeout_ms_(std::max(1, frame_pop_timeout_ms)),
      thread_monitor_enable_(thread_monitor_enable),
      show_thread_status_(show_thread_status),
      enable_pipeline_stats_(enable_pipeline_stats),
      show_stats_1s_(show_stats_1s) {}

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
  constexpr auto kQueueStatsInterval = std::chrono::seconds(1);
  auto next_heartbeat = std::chrono::steady_clock::now();
  auto next_queue_stats_log = next_heartbeat + kQueueStatsInterval;
  FrameQueue::StatsSnapshot last_snapshot{};
  bool has_last_snapshot = false;
  std::uint64_t window_frames = 0U;
  std::uint64_t window_frames_with_roi = 0U;
  std::uint64_t window_frames_with_detection = 0U;
  std::uint64_t window_infer_frames = 0U;
  std::uint64_t window_infer_degraded_frames = 0U;
  std::uint64_t window_infer_cooldown_frames = 0U;
  std::uint64_t window_weak_fallback_frames = 0U;
  std::uint64_t window_raw_candidates = 0U;
  std::uint64_t window_filtered_candidates = 0U;
  std::uint64_t window_nms_candidates = 0U;
  std::uint64_t window_topk_candidates = 0U;
  std::uint64_t window_roi_valid_crops = 0U;
  std::uint64_t window_detections = 0U;
  int window_infer_consecutive_failures_last = 0;

  while (running_.load(std::memory_order_relaxed)) {
    const auto now = std::chrono::steady_clock::now();
    if (thread_handle) {
      if (now >= next_heartbeat) {
        thread_handle->heartbeat();
        next_heartbeat = now + kHeartbeatInterval;
      }
    }
    if (show_stats_1s_ && now >= next_queue_stats_log) {
      const FrameQueue::StatsSnapshot snapshot = frame_queue_->GetStatsSnapshot();
      const std::uint64_t delta_push =
          has_last_snapshot ? (snapshot.push_total - last_snapshot.push_total) : snapshot.push_total;
      const std::uint64_t delta_pop =
          has_last_snapshot ? (snapshot.pop_total - last_snapshot.pop_total) : snapshot.pop_total;
      const std::uint64_t delta_drop_overflow =
          has_last_snapshot ? (snapshot.drop_overflow - last_snapshot.drop_overflow)
                            : snapshot.drop_overflow;
      const std::uint64_t delta_drop_stale =
          has_last_snapshot ? (snapshot.drop_stale - last_snapshot.drop_stale)
                            : snapshot.drop_stale;
      const std::uint64_t delta_drop_total = delta_drop_overflow + delta_drop_stale;
      const double drop_rate =
          (delta_push > 0U)
              ? static_cast<double>(delta_drop_total) / static_cast<double>(delta_push)
              : 0.0;
      const double roi_pass_rate =
          (window_topk_candidates > 0U)
              ? static_cast<double>(window_roi_valid_crops) /
                    static_cast<double>(window_topk_candidates)
              : 0.0;
      const double infer_degraded_rate =
          (window_infer_frames > 0U)
              ? static_cast<double>(window_infer_degraded_frames) /
                    static_cast<double>(window_infer_frames)
              : 0.0;
      const double weak_fallback_rate =
          (window_infer_frames > 0U)
              ? static_cast<double>(window_weak_fallback_frames) /
                    static_cast<double>(window_infer_frames)
              : 0.0;
      const double final_det_rate =
          (window_frames > 0U)
              ? static_cast<double>(window_frames_with_detection) /
                    static_cast<double>(window_frames)
              : 0.0;
      WUST_INFO("detect_worker")
          << "event=pipeline_1s"
          << ", queue_push=" << delta_push
          << ", pop=" << delta_pop
          << ", drop_overflow=" << delta_drop_overflow
          << ", drop_stale=" << delta_drop_stale
          << ", drop_rate=" << drop_rate
          << ", push_total=" << snapshot.push_total
          << ", pop_total=" << snapshot.pop_total
          << ", drop_overflow_total=" << snapshot.drop_overflow
          << ", drop_stale_total=" << snapshot.drop_stale
          << ", queue_size=" << snapshot.queue_size
          << ", capacity=" << snapshot.capacity
          << ", frames=" << window_frames
          << ", raw=" << window_raw_candidates
          << ", filtered=" << window_filtered_candidates
          << ", nms=" << window_nms_candidates
          << ", topk=" << window_topk_candidates
          << ", frames_with_roi=" << window_frames_with_roi
          << ", roi_valid_crops=" << window_roi_valid_crops
          << ", roi_pass_rate=" << roi_pass_rate
          << ", infer_frames=" << window_infer_frames
          << ", infer_degraded_frames=" << window_infer_degraded_frames
          << ", infer_degraded_rate=" << infer_degraded_rate
          << ", infer_cooldown_frames=" << window_infer_cooldown_frames
          << ", infer_consecutive_failures_last=" << window_infer_consecutive_failures_last
          << ", weak_fallback_frames=" << window_weak_fallback_frames
          << ", weak_fallback_rate=" << weak_fallback_rate
          << ", final_detections=" << window_detections
          << ", final_det_rate=" << final_det_rate;
      last_snapshot = snapshot;
      has_last_snapshot = true;
      next_queue_stats_log = now + kQueueStatsInterval;
      window_frames = 0U;
      window_frames_with_roi = 0U;
      window_frames_with_detection = 0U;
      window_infer_frames = 0U;
      window_infer_degraded_frames = 0U;
      window_infer_cooldown_frames = 0U;
      window_weak_fallback_frames = 0U;
      window_raw_candidates = 0U;
      window_filtered_candidates = 0U;
      window_nms_candidates = 0U;
      window_topk_candidates = 0U;
      window_roi_valid_crops = 0U;
      window_detections = 0U;
      window_infer_consecutive_failures_last = 0;
    }

    FramePacket frame;
    if (!frame_queue_->Pop(&frame, frame_pop_timeout_ms_)) {
      continue;
    }

    PipelineFrameStats frame_stats;
    const bool need_frame_stats = enable_pipeline_stats_ || show_stats_1s_;
    std::vector<Detection> detections =
        pipeline_->Process(frame, need_frame_stats ? &frame_stats : nullptr);//逐帧处理
    if (need_frame_stats) {
      ++window_frames;
      window_raw_candidates += frame_stats.raw_candidates;
      window_filtered_candidates += frame_stats.filtered_candidates;
      window_nms_candidates += frame_stats.nms_candidates;
      window_topk_candidates += frame_stats.topk_candidates;
      window_roi_valid_crops += static_cast<std::uint64_t>(std::max(0, frame_stats.roi_valid_crops));
      if (frame_stats.roi_ready) {
        ++window_frames_with_roi;
      }
      if (frame_stats.infer_executed) {
        ++window_infer_frames;
      }
      if (frame_stats.infer_degraded) {
        ++window_infer_degraded_frames;
      }
      if (frame_stats.infer_cooldown_active) {
        ++window_infer_cooldown_frames;
      }
      if (frame_stats.weak_fallback_triggered) {
        ++window_weak_fallback_frames;
      }
      if (frame_stats.final_detections > 0U) {
        ++window_frames_with_detection;
      }
      window_detections += frame_stats.final_detections;
      window_infer_consecutive_failures_last = frame_stats.infer_consecutive_failures;
    }
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
