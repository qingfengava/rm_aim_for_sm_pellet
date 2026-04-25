#include "pellet/imgprocess/morphology.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <wust_vl/common/utils/logger.hpp>

namespace pellet::imgprocess {
namespace {

struct MaskStats {
  int nonzero{0};
  int components{0};
  int largest_component_area{0};
};

//统一输入掩码为二值单通道格式
cv::Mat NormalizeToBinaryMask(const cv::Mat& mask) {
  if (mask.empty()) {
    return {};
  }

  cv::Mat gray;
  if (mask.channels() == 1) {
    gray = mask;
  } else {
    cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
  }

  cv::Mat mask_u8;
  if (gray.type() == CV_8UC1) {
    mask_u8 = gray;
  } else {
    gray.convertTo(mask_u8, CV_8UC1);
  }

  cv::Mat mask_bin;
  cv::threshold(mask_u8, mask_bin, 0, 255, cv::THRESH_BINARY);
  return mask_bin;
}

cv::Mat BuildMorphKernel(int kernel_size) {
  int ksize = kernel_size;
  if (ksize < 1) {
    ksize = 1;
  }
  if ((ksize % 2) == 0) {
    ++ksize;
  }
  return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{ksize, ksize});
}

MaskStats AnalyzeMask(const cv::Mat& mask_bin) {
  MaskStats stats;
  if (mask_bin.empty()) {
    return stats;
  }

  stats.nonzero = cv::countNonZero(mask_bin);
  if (stats.nonzero <= 0) {
    return stats;
  }

  cv::Mat labels;
  cv::Mat cc_stats;
  cv::Mat centroids;
  const int num_labels = cv::connectedComponentsWithStats(mask_bin, labels, cc_stats, centroids, 8, CV_32S);
  stats.components = std::max(0, num_labels - 1);

  for (int label = 1; label < num_labels; ++label) {
    stats.largest_component_area = std::max(stats.largest_component_area, cc_stats.at<int>(label, cv::CC_STAT_AREA));
  }

  return stats;
}

}  // namespace

cv::Mat ApplyOpen(const cv::Mat& binary_mask, int kernel_size, int iterations, bool debug_mode) {
  cv::Mat mask_bin = NormalizeToBinaryMask(binary_mask);
  if (mask_bin.empty()) {
    return {};
  }

  const int iters = std::max(0, iterations);
  if (iters == 0) {
    return mask_bin;
  }

  const cv::Mat kernel = BuildMorphKernel(kernel_size);

  cv::Mat eroded;
  cv::Mat opened;
  cv::erode(mask_bin, eroded, kernel, cv::Point{-1, -1}, iters);
  cv::dilate(eroded, opened, kernel, cv::Point{-1, -1}, iters);

  cv::threshold(opened, opened, 0, 255, cv::THRESH_BINARY);

  //安全回退：当开操作导致前景像素过度丢失时，回退到原始二值掩码。
  const int before_nonzero = cv::countNonZero(mask_bin);
  if (before_nonzero <= 0) {
    return opened;
  }

  const int after_nonzero = cv::countNonZero(opened);
  constexpr double kMinKeepRatio = 0.35;
  const double keep_ratio = static_cast<double>(after_nonzero) / static_cast<double>(before_nonzero);
  if (keep_ratio < kMinKeepRatio) {
    if (debug_mode) {
      WUST_WARN("morphology") << "ApplyOpen fallback triggered: foreground keep ratio too low "
                              << "(before=" << before_nonzero
                              << ", after=" << after_nonzero
                              << ", ratio=" << keep_ratio
                              << ", threshold=" << kMinKeepRatio << ")";
    }
    return mask_bin;
  }

  return opened;
}

cv::Mat ApplyClose(const cv::Mat& binary_mask, int kernel_size, int iterations, bool debug_mode) {
  cv::Mat mask_bin = NormalizeToBinaryMask(binary_mask);
  if (mask_bin.empty()) {
    return {};
  }

  const int iters = std::max(0, iterations);
  if (iters == 0) {
    return mask_bin;
  }

  const cv::Mat kernel = BuildMorphKernel(kernel_size);

  cv::Mat dilated;
  cv::Mat closed;
  cv::dilate(mask_bin, dilated, kernel, cv::Point{-1, -1}, iters);
  cv::erode(dilated, closed, kernel, cv::Point{-1, -1}, iters);

  cv::threshold(closed, closed, 0, 255, cv::THRESH_BINARY);

  //安全回退：当闭操作导致前景像素过度增长或组件过度合并时，回退到原始二值掩码。
  const MaskStats before = AnalyzeMask(mask_bin);
  if (before.nonzero <= 0) {
    return closed;
  }
  const MaskStats after = AnalyzeMask(closed);

  const double r_fg = static_cast<double>(after.nonzero) / static_cast<double>(before.nonzero);
  const double r_cand =
      static_cast<double>(after.components) / static_cast<double>(std::max(1, before.components));
  const double r_largest =
      static_cast<double>(after.largest_component_area) / static_cast<double>(std::max(1, after.nonzero));

  constexpr double kMaxFgGrowthRatio = 2.0;
  constexpr double kMaxLargestComponentRatio = 0.7;
  constexpr double kMinCandidateKeepRatio = 0.5;
  constexpr double kCandidateDropFgGrowthGate = 1.4;

  const bool fallback =
      (r_fg > kMaxFgGrowthRatio) ||
      (r_largest > kMaxLargestComponentRatio) ||
      ((r_cand < kMinCandidateKeepRatio) && (r_fg > kCandidateDropFgGrowthGate));

  if (fallback) {
    if (debug_mode) {
      WUST_WARN("morphology") << "ApplyClose fallback triggered: possible over-merge "
                              << "(fg_before=" << before.nonzero
                              << ", fg_after=" << after.nonzero
                              << ", r_fg=" << r_fg
                              << ", cand_before=" << before.components
                              << ", cand_after=" << after.components
                              << ", r_cand=" << r_cand
                              << ", largest_after=" << after.largest_component_area
                              << ", r_largest=" << r_largest
                              << ")";
    }
    return mask_bin;
  }

  return closed;
}

}  // namespace pellet::imgprocess
