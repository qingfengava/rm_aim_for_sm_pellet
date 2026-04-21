#pragma once

#include <opencv2/core/mat.hpp>

namespace pellet::imgprocess {

class ThreeFrameDiff {
 public:
  ThreeFrameDiff() = default;

  cv::Mat Apply(const cv::Mat& gray_frame, int high_threshold);
  void Reset();

 private:
  cv::Mat prev2_;
  cv::Mat prev1_;
};

}  // namespace pellet::imgprocess
