#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "pellet/config.h"
#include "pellet/infer/i_classifier.h"
#include "pellet/imgprocess/three_frame_diff.h"
#include "pellet/type.h"

namespace pellet::detector {

struct FramePacket {
  uint32_t frame_id{0};
  int64_t timestamp_ms{0};
  cv::Mat frame_bgr;
};

class DetectorPipeline {
 public:
  DetectorPipeline(PelletConfig config, std::shared_ptr<infer::IClassifier> classifier);

  std::vector<Detection> Process(const FramePacket& frame);

 private:
  PelletConfig config_{};
  std::shared_ptr<infer::IClassifier> classifier_;
  imgprocess::ThreeFrameDiff three_frame_diff_;
};

}  // namespace pellet::detector
