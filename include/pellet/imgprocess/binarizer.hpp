#pragma once

#include <opencv2/core/mat.hpp>

namespace pellet::imgprocess {

cv::Mat BinarizeMotion(const cv::Mat& motion_response, int threshold_low, int threshold_high);

}  // namespace pellet::imgprocess
