#include "pellet/detector/capture_worker.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <string>

#include "pellet/utils/time_utils.h"

namespace pellet::detector {
namespace {

bool IsIntegerString(const std::string& text) {
  if (text.empty()) {
    return false;
  }
  return std::all_of(text.begin(), text.end(), [](unsigned char c) {
    return std::isdigit(c) != 0;
  });
}

}  // namespace

CaptureWorker::CaptureWorker(const CameraConfig& config, FrameQueue* frame_queue)
    : config_(config), frame_queue_(frame_queue) {}

CaptureWorker::~CaptureWorker() {
  Stop();
}

bool CaptureWorker::Start() {
  if (running_.load() || frame_queue_ == nullptr) {
    return false;
  }

  if (IsIntegerString(config_.source)) {
    capture_.open(std::stoi(config_.source));
  } else {
    capture_.open(config_.source);
  }

  if (!capture_.isOpened()) {
    return false;
  }

  capture_.set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
  capture_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
  capture_.set(cv::CAP_PROP_FPS, config_.fps);
  if (config_.set_manual_exposure) {
    capture_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1.0);
    capture_.set(cv::CAP_PROP_EXPOSURE, config_.exposure);
  }

  running_.store(true);
  worker_thread_ = std::thread(&CaptureWorker::Run, this);
  return true;
}

void CaptureWorker::Stop() {
  running_.store(false);
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  if (capture_.isOpened()) {
    capture_.release();
  }
}

void CaptureWorker::Run() {
  while (running_.load()) {
    cv::Mat frame;
    if (!capture_.read(frame)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    FramePacket packet;
    packet.frame_id = frame_id_++;
    packet.timestamp_ms = utils::NowMs();
    packet.frame_bgr = frame;

    frame_queue_->Push(packet);
  }
}

}  // namespace pellet::detector
