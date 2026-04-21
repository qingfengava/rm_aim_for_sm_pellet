#include "pellet/detector/detector.h"

#include <memory>
#include <utility>

#include "pellet/detector/capture_worker.h"
#include "pellet/detector/detect_worker.h"
#include "pellet/detector/detector_pipeline.h"
#include "pellet/detector/frame_queue.h"
#include "pellet/infer/i_classifier.h"

namespace pellet {

PelletDetector::PelletDetector(PelletConfig config) : config_(std::move(config)) {}

PelletDetector::~PelletDetector() {
  Stop();
}

bool PelletDetector::Init() {
  if (initialized_) {
    return true;
  }

  classifier_ = infer::CreateClassifier(config_.inference.backend);
  if (!classifier_ || !classifier_->Init(config_.inference)) {
    return false;
  }

  pipeline_ = std::make_unique<detector::DetectorPipeline>(config_, classifier_);
  frame_queue_ = std::make_unique<detector::FrameQueue>(3);
  capture_worker_ = std::make_unique<detector::CaptureWorker>(config_.camera, frame_queue_.get());
  detect_worker_ = std::make_unique<detector::DetectWorker>(frame_queue_.get(), pipeline_.get());

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

bool PelletDetector::PopDetections(std::vector<Detection>* detections, int timeout_ms) {
  if (!detect_worker_) {
    return false;
  }
  return detect_worker_->PopLatest(detections, timeout_ms);
}

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
