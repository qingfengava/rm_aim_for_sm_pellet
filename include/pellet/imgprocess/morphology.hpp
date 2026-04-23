#pragma once

#include <opencv2/core/mat.hpp>

namespace pellet::imgprocess {

cv::Mat ApplyOpen(const cv::Mat& binary_mask, int kernel_size, int iterations, bool debug_mode = false);
cv::Mat ApplyClose(const cv::Mat& binary_mask, int kernel_size, int iterations, bool debug_mode = false);

}  // namespace pellet::imgprocess
