#include "pellet/imgprocess/morphology.h"

#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {

cv::Mat ApplyOpen(const cv::Mat& binary_mask, int kernel_size, int iterations) {
  if (binary_mask.empty()) {
    return {};
  }

  int ksize = kernel_size;
  if (ksize < 1) {
    ksize = 1;
  }
  if ((ksize % 2) == 0) {
    ++ksize;
  }

  cv::Mat opened;
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{ksize, ksize});
  cv::morphologyEx(binary_mask, opened, cv::MORPH_OPEN, kernel, cv::Point{-1, -1}, std::max(1, iterations));
  return opened;
}

}  // namespace pellet::imgprocess
