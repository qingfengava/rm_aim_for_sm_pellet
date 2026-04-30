#include "pellet/imgprocess/preprocess.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {
namespace {

constexpr int kMaxGaussianKernelSize = 31;

int ResolveGaussianKernelSize(int ksize) {
  const int positive = std::clamp(ksize, 1, kMaxGaussianKernelSize);
  return (positive % 2 == 0) ? (positive + 1) : positive;
}

double ResolveGaussianSigma(double sigma) {
  if (!std::isfinite(sigma)) {
    return 0.0;
  }
  return std::max(0.0, sigma);
}

bool ToGrayU8(const cv::Mat& frame_bgr, cv::Mat* gray_u8) {
  if (gray_u8 == nullptr) {
    return false;
  }
  if (frame_bgr.empty()) {
    gray_u8->release();
    return false;
  }

  cv::Mat gray;
  if (frame_bgr.type() == CV_8UC1) {
    *gray_u8 = frame_bgr;
    return true;
  }
  if (frame_bgr.channels() == 1) {
    frame_bgr.convertTo(*gray_u8, CV_8UC1);
    return !gray_u8->empty();
  } else if (frame_bgr.channels() == 3) {
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
  } else if (frame_bgr.channels() == 4) {
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGRA2GRAY);
  } else {
    gray_u8->release();
    return false;
  }

  if (gray.type() == CV_8UC1) {
    *gray_u8 = gray;
    return true;
  }

  gray.convertTo(*gray_u8, CV_8UC1);
  return !gray_u8->empty();
}

}  // namespace

void ToGrayAndBlur(
    const cv::Mat& frame_bgr,
    int gaussian_ksize,
    double gaussian_sigma,
    cv::Mat* output_gray,
    PreprocessScratch* scratch) {
  if (output_gray == nullptr) {
    return;
  }
  output_gray->release();

  cv::Mat gray_local;
  cv::Mat* gray_u8 = scratch != nullptr ? &scratch->gray_u8 : &gray_local;
  if (!ToGrayU8(frame_bgr, gray_u8) || gray_u8->empty()) {
    return;
  }

  const double sigma = ResolveGaussianSigma(gaussian_sigma);
  const int ksize = ResolveGaussianKernelSize(gaussian_ksize);
  if (ksize == 1 && sigma <= 0.0) {
    *output_gray = *gray_u8;
    return;
  }

  if (scratch != nullptr) {
    cv::GaussianBlur(
        *gray_u8,
        scratch->blurred,
        cv::Size{ksize, ksize},
        sigma,
        sigma);
    *output_gray = scratch->blurred;
    return;
  }

  cv::Mat blurred;
  cv::GaussianBlur(
      *gray_u8,
      blurred,
      cv::Size{ksize, ksize},
      sigma,
      sigma);
  *output_gray = blurred;
}

cv::Mat ToGrayAndBlur(const cv::Mat& frame_bgr, int gaussian_ksize, double gaussian_sigma) {
  // 兼容旧接口：内部使用线程局部缓存，降低逐帧分配开销。
  static thread_local PreprocessScratch scratch;
  cv::Mat output_gray;
  ToGrayAndBlur(
      frame_bgr,
      gaussian_ksize,
      gaussian_sigma,
      &output_gray,
      &scratch);
  return output_gray;
}

}  // namespace pellet::imgprocess
