#pragma once

#include <opencv2/core/mat.hpp>

namespace pellet::imgprocess {

struct PreprocessScratch {
  cv::Mat gray_u8;
  cv::Mat blurred;
};

// 统一预处理：转灰度 + 高斯模糊（复用缓存版本）。
void ToGrayAndBlur(
    const cv::Mat& frame_bgr,
    int gaussian_ksize,
    double gaussian_sigma,
    cv::Mat* output_gray,
    PreprocessScratch* scratch);

// 统一预处理：转灰度 + 高斯模糊（便捷返回版本）。
cv::Mat ToGrayAndBlur(const cv::Mat& frame_bgr, int gaussian_ksize, double gaussian_sigma);

}  // namespace pellet::imgprocess
