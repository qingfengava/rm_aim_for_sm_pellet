#pragma once

#include <opencv2/core/mat.hpp>

namespace pellet::imgprocess {

cv::Mat ToGrayAndBlur(const cv::Mat& frame_bgr, int gaussian_ksize, double gaussian_sigma);

}  // namespace pellet::imgprocess
