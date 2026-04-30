#pragma once

#include <memory>
#include <vector>

#include "pellet/config.hpp"
#include "pellet/detector/frame_packet.hpp"
#include "pellet/infer/i_classifier.hpp"
#include "pellet/imgprocess/bg_subtractor.hpp"
#include "pellet/imgprocess/preprocess.hpp"
#include "pellet/type.hpp"

namespace pellet::detector {

struct PipelineFrameStats {
  bool frame_valid{false};
  bool preprocess_ok{false};
  bool motion_ok{false};
  bool roi_ready{false};
  bool infer_executed{false};
  bool infer_degraded{false};
  bool infer_cooldown_active{false};
  bool weak_fallback_triggered{false};
  int infer_consecutive_failures{0};
  std::size_t raw_candidates{0};
  std::size_t filtered_candidates{0};
  std::size_t nms_candidates{0};
  std::size_t topk_candidates{0};
  int roi_total_candidates{0};
  int roi_valid_crops{0};
  int roi_filtered_low_quality{0};
  int roi_filtered_low_texture{0};
  int roi_filtered_oob{0};
  float roi_avg_size{0.0F};
  std::size_t infer_inputs{0};
  std::size_t infer_outputs{0};
  std::size_t final_detections{0};
};

class DetectorPipeline {
 public:
  DetectorPipeline(PelletConfig config, std::shared_ptr<infer::IClassifier> classifier);

  std::vector<Detection> Process(const FramePacket& frame);
  std::vector<Detection> Process(const FramePacket& frame, PipelineFrameStats* stats);

 private:
  PelletConfig config_{};
  std::shared_ptr<infer::IClassifier> classifier_;
  imgprocess::PreprocessScratch preprocess_scratch_{};
  imgprocess::BgSubtractor bg_subtractor_;
};

}  // namespace pellet::detector
