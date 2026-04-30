#include "pellet/imgprocess/roi_cropper.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {
namespace {

constexpr float kQualityLowThreshold = 0.08F;
constexpr float kQualityHighThreshold = 0.22F;
constexpr float kWeakCandidateScale = 1.18F;
constexpr float kStrongCandidateScale = 0.92F;
bool NormalizeToGrayU8(const cv::Mat& src, cv::Mat* dst) {
  if (dst == nullptr || src.empty()) {
    return false;
  }
  if (src.type() == CV_8UC1) {
    *dst = src;
    return true;
  }
  if (src.channels() == 1) {
    src.convertTo(*dst, CV_8UC1);
    return !dst->empty();
  }
  if (src.channels() == 3) {
    cv::cvtColor(src, *dst, cv::COLOR_BGR2GRAY);
    return !dst->empty();
  }
  if (src.channels() == 4) {
    cv::cvtColor(src, *dst, cv::COLOR_BGRA2GRAY);
    return !dst->empty();
  }
  return false;
}

float ComputeQualityScore(const Candidate& candidate, bool* has_quality_signal) {
  const float motion = std::clamp(candidate.motion_score, 0.0F, 1.0F);
  const float contrast = std::clamp(candidate.local_contrast, 0.0F, 1.0F);
  if (has_quality_signal != nullptr) {
    *has_quality_signal = (motion > 0.0F) || (contrast > 0.0F);
  }
  return 0.60F * motion + 0.40F * contrast;
}

cv::Rect BuildSquareRoi(const cv::Point2f& center, int side) {
  const int side_safe = std::max(1, side);
  const int half = side_safe / 2;
  const int cx = static_cast<int>(std::round(center.x));
  const int cy = static_cast<int>(std::round(center.y));
  return cv::Rect{cx - half, cy - half, side_safe, side_safe};
}

cv::Rect ClampRect(const cv::Rect& rect, const cv::Size& bounds) {
  const int x = std::clamp(rect.x, 0, bounds.width);
  const int y = std::clamp(rect.y, 0, bounds.height);
  const int max_width = std::max(0, bounds.width - x);
  const int max_height = std::max(0, bounds.height - y);
  const int width = std::clamp(rect.width, 0, max_width);
  const int height = std::clamp(rect.height, 0, max_height);
  return {x, y, width, height};
}

// 在预分配缓冲区上做补边提取。buffer 必须 >= desired_roi.size()。
// 仅操作 buffer 左上角 desired_roi.size() 区域，其余区域不动。
bool ExtractSquarePatchIntoBuffer(
    const cv::Mat& gray_u8,
    const cv::Rect& desired_roi,
    cv::Mat* buffer,
    cv::Rect* clamped_roi) {
  if (buffer == nullptr || gray_u8.empty()) {
    return false;
  }

  const cv::Rect frame_rect{0, 0, gray_u8.cols, gray_u8.rows};
  const cv::Rect clipped_roi = desired_roi & frame_rect;
  if (clipped_roi.width <= 0 || clipped_roi.height <= 0) {
    return false;
  }
  if (clamped_roi != nullptr) {
    *clamped_roi = clipped_roi;
  }

  cv::Mat view = (*buffer)(cv::Rect(0, 0, desired_roi.width, desired_roi.height));
  view.setTo(0);

  const cv::Rect dst_roi{
      clipped_roi.x - desired_roi.x,
      clipped_roi.y - desired_roi.y,
      clipped_roi.width,
      clipped_roi.height};
  gray_u8(clipped_roi).copyTo(view(dst_roi));
  return true;
}

}  // namespace

RoiBatch CropRoiBatch(
    const cv::Mat& gray_frame,
    const std::vector<Candidate>& candidates,
    const RoiCropConfig& config,
    RoiCropStats* stats) {
  RoiBatch batch;
  RoiCropStats local_stats;

  if (gray_frame.empty() || candidates.empty()) {
    if (stats != nullptr) {
      *stats = local_stats;
    }
    return batch;
  }

  cv::Mat gray_u8;
  if (!NormalizeToGrayU8(gray_frame, &gray_u8) || gray_u8.empty()) {
    if (stats != nullptr) {
      *stats = local_stats;
    }
    return batch;
  }

  const int min_crop = std::max(1, config.min_crop);
  const int max_crop = std::max(min_crop, config.max_crop);
  const int output_size = std::max(1, config.output_size);

  batch.patches.reserve(candidates.size());
  batch.boxes.reserve(candidates.size());
  batch.centers.reserve(candidates.size());

  // 预分配一次，循环内不重新分配
  cv::Mat square_buf(max_crop, max_crop, CV_8UC1);

  for (const auto& candidate : candidates) {
    ++local_stats.total_candidates;

    if (candidate.center.x < 0.0F || candidate.center.x >= static_cast<float>(gray_u8.cols) ||
        candidate.center.y < 0.0F || candidate.center.y >= static_cast<float>(gray_u8.rows)) {
      ++local_stats.filtered_out_of_bounds;
      continue;
    }

    bool has_quality_signal = false;
    const float quality_score = ComputeQualityScore(candidate, &has_quality_signal);
    if (has_quality_signal && quality_score < kQualityLowThreshold) {
      ++local_stats.filtered_low_quality;
      continue;
    }

    float dynamic_scale = 1.0F;
    if (has_quality_signal) {
      dynamic_scale = (quality_score < kQualityHighThreshold)
                          ? kWeakCandidateScale
                          : kStrongCandidateScale;
    }

    const float side_f =
        std::sqrt(static_cast<float>(std::max(1, candidate.area))) *
        config.size_scale *
        dynamic_scale;
    const int side = std::clamp(
        static_cast<int>(std::round(side_f)),
        min_crop,
        max_crop);

    const cv::Rect desired_roi = BuildSquareRoi(candidate.center, side);
    cv::Rect clamped_roi;
    if (!ExtractSquarePatchIntoBuffer(gray_u8, desired_roi, &square_buf, &clamped_roi)) {
      ++local_stats.filtered_out_of_bounds;
      continue;
    }

    const int interpolation = (side > output_size) ? cv::INTER_AREA : cv::INTER_LINEAR;
    const cv::Mat patch_view = square_buf(cv::Rect(0, 0, side, side));
    batch.patches.emplace_back();
    cv::resize(
        patch_view,
        batch.patches.back(),
        cv::Size{output_size, output_size},
        0.0,
        0.0,
        interpolation);
    if (batch.patches.back().empty() || batch.patches.back().type() != CV_8UC1) {
      batch.patches.pop_back();
      continue;
    }

    batch.boxes.push_back(clamped_roi);
    batch.centers.push_back(candidate.center);
    ++local_stats.valid_crops;
    local_stats.avg_crop_size += static_cast<float>(side);
  }

  if (local_stats.valid_crops > 0) {
    local_stats.avg_crop_size /=
        static_cast<float>(local_stats.valid_crops);
  } else {
    local_stats.avg_crop_size = 0.0F;
  }

  if (stats != nullptr) {
    *stats = local_stats;
  }

  return batch;
}

cv::Mat CropSingleRoi(
    const cv::Mat& gray_frame,
    const cv::Rect& roi,
    int output_size,
    bool debug_mode) {
  (void)debug_mode;
  cv::Mat gray_u8;
  if (!NormalizeToGrayU8(gray_frame, &gray_u8) || gray_u8.empty()) {
    return {};
  }
  const int target_size = std::max(1, output_size);
  const cv::Rect clamped = ClampRect(roi, gray_u8.size());
  if (clamped.width <= 0 || clamped.height <= 0) {
    return {};
  }

  cv::Mat resized;
  const int interpolation =
      (clamped.width > target_size || clamped.height > target_size)
          ? cv::INTER_AREA
          : cv::INTER_LINEAR;
  cv::resize(
      gray_u8(clamped),
      resized,
      cv::Size{target_size, target_size},
      0.0,
      0.0,
      interpolation);
  return resized;
}

}  // namespace pellet::imgprocess
