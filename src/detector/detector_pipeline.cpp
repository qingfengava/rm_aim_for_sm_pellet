#include "pellet/detector/detector_pipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <vector>

#include <opencv2/core/types.hpp>

#include "pellet/imgprocess/binarizer.h"
#include "pellet/imgprocess/candidate_extractor.h"
#include "pellet/imgprocess/candidate_filter.h"
#include "pellet/imgprocess/candidate_nms.h"
#include "pellet/imgprocess/morphology.h"
#include "pellet/imgprocess/preprocess.h"
#include "pellet/imgprocess/roi_cropper.h"

namespace pellet::detector {
namespace {

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

constexpr int kMaxPreNmsCandidates = 20;

}  // namespace

DetectorPipeline::DetectorPipeline(PelletConfig config, std::shared_ptr<infer::IClassifier> classifier)
    : config_(std::move(config)), classifier_(std::move(classifier)) {}


//pipline:  preprocess -> three_frame_diff -> morphology ->  
//  candidate extract -> filter & rank -> nms -> topk -> classifier
std::vector<Detection> DetectorPipeline::Process(const FramePacket& frame) {
  std::vector<Detection> detections;
  if (frame.frame_bgr.empty()) {
    return detections;
  }

  const cv::Mat gray = imgprocess::ToGrayAndBlur(
      frame.frame_bgr,
      config_.motion.gaussian_ksize,
      config_.motion.gaussian_sigma);

  const cv::Mat motion_response =
      three_frame_diff_.Apply(gray, config_.motion.diff_threshold_max);
  const cv::Mat binary = imgprocess::BinarizeMotion(
      motion_response,
      config_.motion.diff_threshold_min,
      config_.motion.diff_threshold);

  cv::Mat mask = binary;
  if (config_.motion.morph_enable) {
    const std::string morph_type = ToLower(config_.motion.morph_type);
    const bool morph_debug = config_.debug.show_morphology;
    if (morph_type == "open") {
      mask = imgprocess::ApplyOpen(
          binary,
          config_.motion.morph_kernel,
          config_.motion.morph_iters,
          morph_debug);
    } else if (morph_type == "close") {
      mask = imgprocess::ApplyClose(
          binary,
          config_.motion.morph_kernel,
          config_.motion.morph_iters,
          morph_debug);
    }
  }

  const std::vector<imgprocess::Candidate> raw_candidates =
      imgprocess::ExtractCandidates(mask, gray, motion_response);

  imgprocess::CandidateFilterConfig filter_cfg;
  filter_cfg.area_min = config_.motion.area_min;
  filter_cfg.area_max = config_.motion.area_max;
  filter_cfg.aspect_ratio_max = config_.motion.ratio_max;
  filter_cfg.extent_min = config_.motion.extent_min;
  filter_cfg.contrast_min = config_.motion.contrast_min;
  filter_cfg.motion_score_min = config_.motion.motion_score_min;
  filter_cfg.max_candidates = std::clamp(config_.motion.max_candidates, 0, kMaxPreNmsCandidates);

  const std::vector<imgprocess::Candidate> filtered_candidates =
      imgprocess::FilterAndRankCandidates(raw_candidates, filter_cfg);

  std::vector<imgprocess::Candidate> nms_candidates = filtered_candidates;
  if (config_.motion.nms_enable) {
    const float nms_iou = std::clamp(config_.motion.nms_iou, 0.0F, 1.0F);
    nms_candidates = imgprocess::ApplyNms(filtered_candidates, nms_iou);
  }

  std::vector<imgprocess::Candidate> topk_candidates = nms_candidates;
  const std::size_t post_nms_topk = static_cast<std::size_t>(std::max(0, config_.inference.max_candidates));
  if (topk_candidates.size() > post_nms_topk) {
    topk_candidates.resize(post_nms_topk);
  }

  if (config_.debug.show_pipeline_stats) {
    const auto now = std::chrono::steady_clock::now();
    if (!stats_log_initialized_ ||
        (now - last_stats_log_tp_) >= std::chrono::seconds(1)) {
      std::cout
          << "[pipeline] candidates raw=" << raw_candidates.size()
          << " -> filtered=" << filtered_candidates.size()
          << " -> nms=" << nms_candidates.size()
          << " -> topk=" << topk_candidates.size()
          << "\n";
      last_stats_log_tp_ = now;
      stats_log_initialized_ = true;
    }
  }

  imgprocess::RoiCropConfig roi_cfg;
  roi_cfg.output_size = config_.roi.output_size;
  roi_cfg.size_scale = config_.roi.size_scale;
  roi_cfg.min_crop = config_.roi.min_crop;
  roi_cfg.max_crop = config_.roi.max_crop;

  const imgprocess::RoiBatch batch = imgprocess::CropRoiBatch(gray, topk_candidates, roi_cfg);
  if (batch.patches.empty()) {
    return detections;
  }

  const std::vector<float> scores = classifier_ != nullptr
                                        ? classifier_->Infer(batch.patches)
                                        : std::vector<float>(batch.patches.size(), 0.0F);

  const std::size_t count = std::min(batch.patches.size(), scores.size());
  detections.reserve(count);

  for (std::size_t i = 0; i < count; ++i) {
    if (scores[i] < config_.inference.positive_threshold) {
      continue;
    }

    Detection det;
    det.frame_id = frame.frame_id;
    det.timestamp_ms = frame.timestamp_ms;
    det.center = batch.centers[i];
    det.bbox = cv::Rect2f(
        static_cast<float>(batch.boxes[i].x),
        static_cast<float>(batch.boxes[i].y),
        static_cast<float>(batch.boxes[i].width),
        static_cast<float>(batch.boxes[i].height));
    det.score = scores[i];
    detections.push_back(det);
  }

  return detections;
}

}  // namespace pellet::detector
