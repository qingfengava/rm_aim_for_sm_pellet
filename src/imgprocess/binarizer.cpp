#include "pellet/imgprocess/binarizer.h"

#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {

cv::Mat BinarizeMotion(const cv::Mat& motion_response, int threshold_low, int threshold_high) {
  if (motion_response.empty()) {
    return {};
  }

  cv::Mat weak;
  cv::threshold(motion_response, weak, threshold_low, 255, cv::THRESH_BINARY);

  if (threshold_high <= threshold_low) {
    return weak;
  }

  cv::Mat strong;
  cv::Mat strong_dilated;
  cv::Mat linked;
  cv::Mat result;

  cv::threshold(motion_response, strong, threshold_high, 255, cv::THRESH_BINARY);
  cv::dilate(strong, strong_dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size{3, 3}));
  cv::bitwise_and(weak, strong_dilated, linked);
  cv::bitwise_or(linked, strong, result);
  return result;
}

}  // namespace pellet::imgprocess
