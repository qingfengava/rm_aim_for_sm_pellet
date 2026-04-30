#pragma once

#include <opencv2/core/mat.hpp>

namespace pellet::imgprocess {

enum class MorphFallbackReason {
  kNone = 0,
  kOpenKeepRatioTooLow = 1,
  kCloseOverMerge = 2,
};

struct MorphologyStats {
  MorphFallbackReason fallback_reason{MorphFallbackReason::kNone};
  int before_nonzero{0};
  int after_nonzero{0};
  int before_components{0};
  int after_components{0};
  double metric_primary{0.0};
  double metric_secondary{0.0};
};

cv::Mat ApplyOpen(
    const cv::Mat& binary_mask,
    int kernel_size,
    int iterations,
    bool debug_mode = false,
    MorphologyStats* stats = nullptr);
cv::Mat ApplyClose(
    const cv::Mat& binary_mask,
    int kernel_size,
    int iterations,
    bool debug_mode = false,
    MorphologyStats* stats = nullptr);

}  // namespace pellet::imgprocess
