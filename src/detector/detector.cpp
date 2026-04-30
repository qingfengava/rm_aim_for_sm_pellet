#include "pellet/detector/detector.hpp"

#include <memory>
#include <utility>

#include "pellet/detector/capture_worker.hpp"
#include "pellet/detector/detect_worker.hpp"
#include "pellet/detector/detector_pipeline.hpp"
#include "pellet/detector/frame_queue.hpp"
#include "pellet/infer/i_classifier.hpp"
#include "pellet/utils/debug_utils.hpp"

namespace pellet {

PelletDetector::PelletDetector(PelletConfig config) : config_(std::move(config)) {}

PelletDetector::~PelletDetector() {
  Stop();
}

bool PelletDetector::Init() {
  if (initialized_) {
    return true;
  }

  infer::InferRuntimeOptions infer_runtime_options;
  infer_runtime_options.debug_log_init =
      utils::IsDebugEnabled(config_, utils::DebugFeature::kInferLogs);
  classifier_ = infer::CreateClassifier(config_.inference.backend);
  if (!classifier_ ||
      !classifier_->Init(config_.inference, infer_runtime_options)) {
    return false;
  }

  const int queue_capacity = config_.detector.queue_capacity > 0
      ? config_.detector.queue_capacity
      : 1;
  const bool capture_debug =
      utils::IsDebugEnabled(config_, utils::DebugFeature::kCaptureLogs);
  const bool thread_status_debug =
      utils::IsDebugEnabled(config_, utils::DebugFeature::kThreadStatus);
  const bool pipeline_stats_debug =
      utils::IsDebugEnabled(config_, utils::DebugFeature::kPipelineStats);
  const bool stats_1s_debug =
      utils::IsDebugEnabled(config_, utils::DebugFeature::kStats1s);
  pipeline_ = std::make_unique<detector::DetectorPipeline>(config_, classifier_);
  frame_queue_ = std::make_unique<detector::FrameQueue>(
      static_cast<std::size_t>(queue_capacity),
      config_.detector.queue_valid_ms,
      config_.detector.pop_poll_ms);
  capture_worker_ = std::make_unique<detector::CaptureWorker>(
      config_.camera, frame_queue_.get(), capture_debug);
  detect_worker_ = std::make_unique<detector::DetectWorker>(
      frame_queue_.get(),
      pipeline_.get(),
      config_.detector.detect_pop_timeout_ms,
      config_.detector.thread_monitor_enable,
      thread_status_debug,
      pipeline_stats_debug,
      stats_1s_debug);

  initialized_ = true;
  return true;
}

bool PelletDetector::Start() {
  if (!Init()) {
    return false;
  }

  detect_worker_->Start();
  if (!capture_worker_->Start()) {
    detect_worker_->Stop();
    return false;
  }
  return true;
}

//stop order: capture_worker->Stop() -> frame_queue->Stop() -> detect_worker->Stop()
void PelletDetector::Stop() {

  if (capture_worker_) {
    capture_worker_->Stop();
  }
  if (frame_queue_) {
    frame_queue_->Stop();
  }
  if (detect_worker_) {
    detect_worker_->Stop();
  }
}

//异步
bool PelletDetector::PopDetections(std::vector<Detection>* detections, int timeout_ms) {
  if (!detect_worker_) {
    return false;
  }
  return detect_worker_->PopLatest(detections, timeout_ms);
}

//同步单帧
std::vector<Detection> PelletDetector::ProcessFrame(
    const cv::Mat& frame_bgr,
    uint32_t frame_id,
    int64_t timestamp_ms) {
  if (!pipeline_) {
    if (!Init()) {
      return {};
    }
  }

  detector::FramePacket packet;
  packet.frame_id = frame_id;
  packet.timestamp_ms = timestamp_ms;
  packet.frame_bgr = frame_bgr;
  return pipeline_->Process(packet);
}

}  // namespace pellet
