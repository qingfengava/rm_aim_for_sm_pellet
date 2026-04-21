#include "pellet/imgprocess/preprocess.h"

#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {

cv::Mat ToGrayAndBlur(const cv::Mat& frame_bgr, int gaussian_ksize, double gaussian_sigma) {
  if (frame_bgr.empty()) {
    return {};
  }

  cv::Mat gray;
  if (frame_bgr.channels() == 1) {
    gray = frame_bgr.clone();
  } else {
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
  }

  int ksize = gaussian_ksize;
  if (ksize < 1) {
    ksize = 1;
  }
  if ((ksize % 2) == 0) {
    ++ksize;
  }

  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size{ksize, ksize}, gaussian_sigma);
  return blurred;
}

}  // namespace pellet::imgprocess
