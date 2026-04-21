#include "pellet/imgprocess/three_frame_diff.h"

#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {

cv::Mat ThreeFrameDiff::Apply(const cv::Mat& gray_frame, int high_threshold) {
  if (gray_frame.empty()) {
    return {};
  }

  if (prev1_.empty()) {
    prev1_ = gray_frame.clone();
    return cv::Mat::zeros(gray_frame.size(), CV_8UC1);
  }

  if (prev2_.empty()) {
    prev2_ = prev1_.clone();
    prev1_ = gray_frame.clone();
    return cv::Mat::zeros(gray_frame.size(), CV_8UC1);
  }

  cv::Mat d1;
  cv::Mat d2;
  cv::Mat and_mask;
  cv::Mat high_mask;
  cv::Mat combined;

  cv::absdiff(prev2_, prev1_, d1);
  cv::absdiff(prev1_, gray_frame, d2);
  cv::bitwise_and(d1, d2, and_mask);
  cv::threshold(d1, high_mask, high_threshold, 255, cv::THRESH_BINARY);
  cv::max(and_mask, high_mask, combined);

  prev2_ = prev1_.clone();
  prev1_ = gray_frame.clone();
  return combined;
}

void ThreeFrameDiff::Reset() {
  prev2_.release();
  prev1_.release();
}

}  // namespace pellet::imgprocess
