#include "pellet/detector/detector_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <wust_vl/common/utils/logger.hpp>

#include "pellet/imgprocess/candidate_extractor.hpp"
#include "pellet/imgprocess/candidate_filter.hpp"
#include "pellet/imgprocess/candidate_nms.hpp"
#include "pellet/imgprocess/morphology.hpp"
#include "pellet/imgprocess/motion_confirm.hpp"
#include "pellet/imgprocess/preprocess.hpp"
#include "pellet/imgprocess/roi_cropper.hpp"
#include "pellet/utils/debug_utils.hpp"

namespace pellet::detector {
namespace {

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

constexpr int kMaxPreNmsCandidates = 8;

struct ScoreThresholds {
  float positive{0.0F};
  float weak{0.0F};
};

ScoreThresholds ResolveScoreThresholds(const InferenceConfig& inference) {
  const float positive = std::clamp(inference.positive_threshold, 0.0F, 1.0F);
  const float weak_raw = std::clamp(inference.weak_threshold, 0.0F, 1.0F);
  const float weak = std::min(weak_raw, positive);
  return ScoreThresholds{positive, weak};
}

const char* MorphFallbackReasonToText(const imgprocess::MorphFallbackReason reason) {
  switch (reason) {
    case imgprocess::MorphFallbackReason::kOpenKeepRatioTooLow:
      return "open_keep_ratio_low";
    case imgprocess::MorphFallbackReason::kCloseOverMerge:
      return "close_over_merge";
    case imgprocess::MorphFallbackReason::kNone:
    default:
      return "none";
  }
}

imgprocess::BgSubtractConfig BuildBgSubtractConfig(const MotionConfig& motion) {
  imgprocess::BgSubtractConfig cfg;
  cfg.backend = motion.bg_backend;
  cfg.history = motion.bg_history;
  cfg.var_threshold = motion.bg_var_threshold;
  cfg.learning_rate = motion.bg_learning_rate;
  cfg.downsample = motion.bg_downsample;
  cfg.detect_shadows = false;
  return cfg;
}

}  // namespace

DetectorPipeline::DetectorPipeline(PelletConfig config,
                                   std::shared_ptr<infer::IClassifier> classifier)
    : config_(std::move(config)),
      classifier_(std::move(classifier)),
      bg_subtractor_(BuildBgSubtractConfig(config_.motion)) {}

std::vector<Detection> DetectorPipeline::Process(const FramePacket& frame) {
  return Process(frame, nullptr);
}

std::vector<Detection> DetectorPipeline::Process(
    const FramePacket& frame,
    PipelineFrameStats* frame_stats) {
  if (frame_stats != nullptr) {
    *frame_stats = PipelineFrameStats{};
  }
  std::vector<Detection> detections;
  if (frame.frame_bgr.empty()) {
    return detections;
  }
  if (frame_stats != nullptr) {
    frame_stats->frame_valid = true;
  }

  // ===== 1. 预处理 =====
  cv::Mat gray;
  imgprocess::ToGrayAndBlur(
      frame.frame_bgr,
      config_.motion.gaussian_ksize,
      config_.motion.gaussian_sigma,
      &gray,
      &preprocess_scratch_);
  if (gray.empty()) {
    return detections;
  }
  if (frame_stats != nullptr) {
    frame_stats->preprocess_ok = true;
  }

  // ===== 2. 运动检测 =====
  imgprocess::BgSubtractResult bg_result = bg_subtractor_.Apply(gray);
  cv::Mat binary = bg_result.binary_mask;
  cv::Mat motion_resp = bg_result.motion_response;

  // ===== 3. 全局扰动保护 =====
  if (!binary.empty()) {
    const int fg_pixels = cv::countNonZero(binary);
    const double fg_ratio =
        static_cast<double>(fg_pixels) / static_cast<double>(binary.total());

    const double hard_drop = config_.motion.global_fg_ratio_max + 0.2;
    if (fg_ratio > hard_drop) {
      bg_subtractor_.SetLearningRate(0.0);
      binary.setTo(0);
      motion_resp.setTo(0);
    } else if (fg_ratio > config_.motion.global_fg_ratio_max) {
      bg_subtractor_.SetLearningRate(0.0);
      motion_resp.convertTo(motion_resp, -1,
                            config_.motion.global_response_attenuation, 0.0);
    } else {
      bg_subtractor_.SetLearningRate(config_.motion.bg_learning_rate);
    }
  }

  if (binary.empty()) {
    return detections;
  }
  if (frame_stats != nullptr) {
    frame_stats->motion_ok = true;
  }

  // ===== 4. 形态学 =====
  cv::Mat mask = binary;
  imgprocess::MorphologyStats morph_stats;
  if (config_.motion.morph_enable) {
    const std::string morph_type = ToLower(config_.motion.morph_type);
    const bool morph_debug = false;
    if (morph_type == "open") {
      mask = imgprocess::ApplyOpen(
          binary,
          config_.motion.morph_kernel,
          config_.motion.morph_iters,
          morph_debug,
          &morph_stats);
    } else if (morph_type == "close") {
      mask = imgprocess::ApplyClose(
          binary,
          config_.motion.morph_kernel,
          config_.motion.morph_iters,
          morph_debug,
          &morph_stats);
    }
    if (morph_stats.fallback_reason != imgprocess::MorphFallbackReason::kNone &&
        utils::ShouldLogRateLimited("detector_pipeline", "morphology_fallback")) {
      WUST_WARN("detector_pipeline")
          << "morphology fallback, type=" << morph_type
          << ", reason=" << MorphFallbackReasonToText(morph_stats.fallback_reason)
          << ", fg_before=" << morph_stats.before_nonzero
          << ", fg_after=" << morph_stats.after_nonzero
          << ", comp_before=" << morph_stats.before_components
          << ", comp_after=" << morph_stats.after_components
          << ", metric_primary=" << morph_stats.metric_primary
          << ", metric_secondary=" << morph_stats.metric_secondary;
    }
  }

  // ===== 5. 候选提取 =====
  std::vector<imgprocess::Candidate> candidates =
      imgprocess::ExtractCandidates(mask, gray, motion_resp);
  if (frame_stats != nullptr) {
    frame_stats->raw_candidates = candidates.size();
  }

  // ===== 6. 运动确认 =====
  if (config_.motion.motion_confirm_enable) {
    candidates = imgprocess::FilterByMotionConfirm(
        candidates, motion_resp, config_.motion.motion_confirm_threshold);
  }

  // ===== 7. 过滤 & 排序 =====
  imgprocess::CandidateFilterConfig filter_cfg;
  filter_cfg.area_min = config_.motion.area_min;
  filter_cfg.area_max = config_.motion.area_max;
  filter_cfg.aspect_ratio_max = config_.motion.ratio_max;
  filter_cfg.extent_min = config_.motion.extent_min;
  filter_cfg.contrast_min = config_.motion.contrast_min;
  filter_cfg.motion_score_min = config_.motion.motion_score_min;
  filter_cfg.max_candidates =
      std::clamp(config_.motion.max_candidates, 0, kMaxPreNmsCandidates);

  candidates = imgprocess::FilterAndRankCandidates(candidates, filter_cfg);
  if (frame_stats != nullptr) {
    frame_stats->filtered_candidates = candidates.size();
  }

  // ===== 8. NMS =====
  if (config_.motion.nms_enable) {
    const float nms_iou = std::clamp(config_.motion.nms_iou, 0.0F, 1.0F);
    candidates = imgprocess::ApplyNmsPreSorted(candidates, nms_iou);
  }
  if (frame_stats != nullptr) {
    frame_stats->nms_candidates = candidates.size();
  }

  // ===== 9. TopK =====
  const std::size_t post_nms_topk =
      static_cast<std::size_t>(std::max(0, config_.inference.max_candidates));
  if (candidates.size() > post_nms_topk) {
    candidates.resize(post_nms_topk);
  }
  if (frame_stats != nullptr) {
    frame_stats->topk_candidates = candidates.size();
  }

  // ===== 10. ROI 裁剪 =====
  imgprocess::RoiCropConfig roi_cfg;
  roi_cfg.output_size = config_.roi.output_size;
  roi_cfg.size_scale = config_.roi.size_scale;
  roi_cfg.min_crop = config_.roi.min_crop;
  roi_cfg.max_crop = config_.roi.max_crop;

  imgprocess::RoiCropStats roi_stats;
  const imgprocess::RoiBatch batch =
      imgprocess::CropRoiBatch(gray, candidates, roi_cfg, &roi_stats);
  if (frame_stats != nullptr) {
    frame_stats->roi_total_candidates = roi_stats.total_candidates;
    frame_stats->roi_valid_crops = roi_stats.valid_crops;
    frame_stats->roi_filtered_low_quality = roi_stats.filtered_low_quality;
    frame_stats->roi_filtered_low_texture = roi_stats.filtered_low_texture;
    frame_stats->roi_filtered_oob = roi_stats.filtered_out_of_bounds;
    frame_stats->roi_avg_size = roi_stats.avg_crop_size;
  }
  if (!candidates.empty() && batch.patches.empty()) {
    if (utils::ShouldLogRateLimited("detector_pipeline", "roi_drop_all")) {
      WUST_WARN("detector_pipeline")
          << "roi crop dropped all candidates, topk=" << candidates.size()
          << ", total_candidates=" << roi_stats.total_candidates
          << ", filtered_low_quality=" << roi_stats.filtered_low_quality
          << ", filtered_low_texture=" << roi_stats.filtered_low_texture
          << ", filtered_out_of_bounds=" << roi_stats.filtered_out_of_bounds;
    }
  }
  if (batch.patches.empty()) {
    if (frame_stats != nullptr) {
      frame_stats->roi_ready = false;
    }
    return detections;
  }
  if (frame_stats != nullptr) {
    frame_stats->roi_ready = true;
    frame_stats->infer_inputs = batch.patches.size();
  }

  // ===== 11. 分类器 =====
  const std::vector<float> scores = classifier_ != nullptr
                                        ? classifier_->Infer(batch.patches)
                                        : std::vector<float>(batch.patches.size(), 0.0F);
  if (frame_stats != nullptr) {
    frame_stats->infer_executed = (classifier_ != nullptr);
    frame_stats->infer_outputs = scores.size();
    if (classifier_ != nullptr) {
      const infer::InferRuntimeState runtime_state = classifier_->GetRuntimeState();
      frame_stats->infer_degraded = runtime_state.degraded_last_call;
      frame_stats->infer_cooldown_active = runtime_state.cooldown_active;
      frame_stats->infer_consecutive_failures = runtime_state.consecutive_failures;
    }
  }

  const std::size_t count = std::min(batch.patches.size(), scores.size());
  const ScoreThresholds score_thresholds = ResolveScoreThresholds(config_.inference);
  detections.reserve(count);
  std::size_t best_weak_index = count;
  float best_weak_score = -1.0F;

  for (std::size_t i = 0; i < count; ++i) {
    if (scores[i] < score_thresholds.positive) {
      if (scores[i] >= score_thresholds.weak && scores[i] > best_weak_score) {
        best_weak_score = scores[i];
        best_weak_index = i;
      }
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

  if (detections.empty() && best_weak_index < count) {
    Detection det;
    det.frame_id = frame.frame_id;
    det.timestamp_ms = frame.timestamp_ms;
    det.center = batch.centers[best_weak_index];
    det.bbox = cv::Rect2f(
        static_cast<float>(batch.boxes[best_weak_index].x),
        static_cast<float>(batch.boxes[best_weak_index].y),
        static_cast<float>(batch.boxes[best_weak_index].width),
        static_cast<float>(batch.boxes[best_weak_index].height));
    det.score = scores[best_weak_index];
    detections.push_back(det);
    if (frame_stats != nullptr) {
      frame_stats->weak_fallback_triggered = true;
    }
  }

  if (frame_stats != nullptr) {
    frame_stats->final_detections = detections.size();
  }

  return detections;
}

}  // namespace pellet::detector
