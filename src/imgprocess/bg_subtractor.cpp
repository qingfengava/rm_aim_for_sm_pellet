#include "pellet/imgprocess/bg_subtractor.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

namespace pellet::imgprocess {

struct BgSubtractor::Impl {
  BgSubtractConfig config;
  cv::Ptr<cv::BackgroundSubtractor> bg;
  cv::Mat prev_gray;
};

BgSubtractor::BgSubtractor(const BgSubtractConfig& config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = config;
  Reset();
}

BgSubtractor::~BgSubtractor() = default;

void BgSubtractor::Reset() {
  if (impl_->config.backend == "mog2") {
    impl_->bg = cv::createBackgroundSubtractorMOG2(
        impl_->config.history,
        impl_->config.var_threshold,
        impl_->config.detect_shadows);
  } else {
    impl_->bg = cv::createBackgroundSubtractorKNN(
        impl_->config.history,
        impl_->config.var_threshold,
        impl_->config.detect_shadows);
  }
  impl_->prev_gray.release();
}

void BgSubtractor::SetLearningRate(double rate) {
  impl_->config.learning_rate = rate;
}

BgSubtractResult BgSubtractor::Apply(const cv::Mat& gray_frame) {
  BgSubtractResult result;
  if (gray_frame.empty()) {
    return result;
  }

  const int ds = std::clamp(impl_->config.downsample, 1, 4);
  cv::Mat work_gray;
  const cv::Size original_size = gray_frame.size();

  if (ds > 1) {
    cv::resize(gray_frame, work_gray,
               cv::Size(gray_frame.cols / ds, gray_frame.rows / ds),
               0.0, 0.0, cv::INTER_LINEAR);
  } else {
    work_gray = gray_frame;
  }

  cv::Mat fg_low;
  impl_->bg->apply(work_gray, fg_low, impl_->config.learning_rate);

  if (impl_->config.detect_shadows) {
    cv::threshold(fg_low, fg_low, 200, 255, cv::THRESH_BINARY);
  }

  if (ds > 1) {
    cv::resize(fg_low, result.binary_mask, original_size,
               0.0, 0.0, cv::INTER_NEAREST);
  } else {
    result.binary_mask = fg_low;
  }

  if (!impl_->prev_gray.empty() && impl_->prev_gray.size() == gray_frame.size()) {
    cv::absdiff(impl_->prev_gray, gray_frame, result.motion_response);
  } else {
    result.motion_response = cv::Mat::zeros(gray_frame.size(), CV_8UC1);
  }

  gray_frame.copyTo(impl_->prev_gray);
  return result;
}

}  // namespace pellet::imgprocess
