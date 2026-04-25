#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "pellet/config.hpp"
#include "pellet/detector/frame_packet.hpp"
#include "pellet/infer/i_classifier.hpp"
#include "pellet/imgprocess/three_frame_diff.hpp"
#include "pellet/type.hpp"

namespace pellet::detector {

class DetectorPipeline {
 public:
  DetectorPipeline(PelletConfig config, std::shared_ptr<infer::IClassifier> classifier);

  std::vector<Detection> Process(const FramePacket& frame);

 private:
  PelletConfig config_{};
  std::shared_ptr<infer::IClassifier> classifier_;
  imgprocess::ThreeFrameDiff three_frame_diff_;
  std::chrono::steady_clock::time_point last_stats_log_tp_{};
  bool stats_log_initialized_{false};
};

}  // namespace pellet::detector
