#include "pellet/detector/detector_pipeline.h"

#include <algorithm>
#include <vector>

#include <opencv2/core/types.hpp>

#include "pellet/imgprocess/binarizer.h"
#include "pellet/imgprocess/candidate_extractor.h"
#include "pellet/imgprocess/candidate_filter.h"
#include "pellet/imgprocess/morphology.h"
#include "pellet/imgprocess/preprocess.h"
#include "pellet/imgprocess/roi_cropper.h"

namespace pellet::detector {

DetectorPipeline::DetectorPipeline(PelletConfig config, std::shared_ptr<infer::IClassifier> classifier)
    : config_(std::move(config)), classifier_(std::move(classifier)) {}

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
  const cv::Mat mask = imgprocess::ApplyOpen(
      binary,
      config_.motion.morph_kernel,
      config_.motion.morph_iters);

  const std::vector<imgprocess::Candidate> raw_candidates =
      imgprocess::ExtractCandidates(mask, gray, motion_response);

  imgprocess::CandidateFilterConfig filter_cfg;
  filter_cfg.area_min = config_.motion.area_min;
  filter_cfg.area_max = config_.motion.area_max;
  filter_cfg.aspect_ratio_max = config_.motion.ratio_max;
  filter_cfg.max_candidates = std::min(config_.motion.max_candidates, config_.inference.max_candidates);

  const std::vector<imgprocess::Candidate> candidates =
      imgprocess::FilterAndRankCandidates(raw_candidates, filter_cfg);

  imgprocess::RoiCropConfig roi_cfg;
  roi_cfg.output_size = config_.roi.output_size;
  roi_cfg.size_scale = config_.roi.size_scale;
  roi_cfg.min_crop = config_.roi.min_crop;
  roi_cfg.max_crop = config_.roi.max_crop;

  const imgprocess::RoiBatch batch = imgprocess::CropRoiBatch(gray, candidates, roi_cfg);
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
