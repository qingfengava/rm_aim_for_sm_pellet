#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "pellet/config.hpp"
#include "pellet/type.hpp"

namespace pellet::detector {
class CaptureWorker;
class DetectWorker;
class DetectorPipeline;
class FrameQueue;
}  // namespace pellet::detector

namespace pellet::infer {
class IClassifier;
}  // namespace pellet::infer

namespace pellet {

class PelletDetector {
 public:
  explicit PelletDetector(PelletConfig config = {});
  ~PelletDetector();

  bool Init();
  bool Start();
  void Stop();

  bool PopDetections(std::vector<Detection>* detections, int timeout_ms);
  std::vector<Detection> ProcessFrame(const cv::Mat& frame_bgr, uint32_t frame_id, int64_t timestamp_ms);

 private:
  PelletConfig config_{};
  bool initialized_{false};

  std::shared_ptr<infer::IClassifier> classifier_;
  std::unique_ptr<detector::DetectorPipeline> pipeline_;
  std::unique_ptr<detector::FrameQueue> frame_queue_;
  std::unique_ptr<detector::CaptureWorker> capture_worker_;
  std::unique_ptr<detector::DetectWorker> detect_worker_;
};

}  // namespace pellet
