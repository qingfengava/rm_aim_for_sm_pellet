#include "pellet/imgprocess/three_frame_diff.h"

#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {

ThreeFrameDiff::ThreeFrameDiff(const ThreeFrameDiffConfig& config)
    : config_(config) {}

cv::Mat ThreeFrameDiff::Apply(const cv::Mat& gray_frame) {
  if (gray_frame.empty()) {
    return {};
  }

  // 初始化
  if (prev1_.empty()) {
    prev1_ = gray_frame.clone();
    return cv::Mat::zeros(gray_frame.size(), CV_8UC1);
  }
  if (prev2_.empty()) {
    prev2_ = prev1_.clone();
    prev1_ = gray_frame.clone();
    return cv::Mat::zeros(gray_frame.size(), CV_8UC1);
  }

  // 三帧差计算
  cv::Mat d1, d2, and_mask;
  cv::absdiff(prev2_, prev1_, d1);
  cv::absdiff(prev1_, gray_frame, d2);
  cv::bitwise_and(d1, d2, and_mask);

  // 自适应计算双阈值
  int t_low, t_high;
  ComputeAdaptiveThresholds(and_mask, t_low, t_high);

  // 高阈值增强
  cv::Mat high_mask;
  //如果噪声太多就改d1->and_mask
  cv::threshold(d1, high_mask, t_high, 255, cv::THRESH_BINARY);

  // 滞回连接（低阈值 + 高阈值种子）
  cv::Mat weak, strong_dilated, linked, result;
  cv::threshold(and_mask, weak, t_low, 255, cv::THRESH_BINARY);
  cv::dilate(high_mask, strong_dilated, 
             cv::getStructuringElement(cv::MORPH_RECT, cv::Size{3, 3}));
  cv::bitwise_and(weak, strong_dilated, linked);
  cv::bitwise_or(linked, high_mask, result);

  // 更新帧缓存
  prev2_ = prev1_.clone();
  prev1_ = gray_frame.clone();

  return result;
}

void ThreeFrameDiff::ComputeAdaptiveThresholds(const cv::Mat& motion_response,
                                               int& t_low, int& t_high) {
  // 检查是否有运动
  double min_val, max_val;
  cv::minMaxLoc(motion_response, &min_val, &max_val);
  
  if (max_val < 1e-6) {
    t_low = config_.static_t_low;
    t_high = config_.static_t_high;
    return;
  }
  
  // 计算均值和标准差
  cv::Scalar mean, stddev;
  cv::meanStdDev(motion_response, mean, stddev);
  
  // 计算双阈值
  t_low = std::clamp(
      static_cast<int>(mean[0] + stddev[0] * config_.adapt_ratio_low),
      config_.min_t_low, config_.max_t_low);
  t_high = std::clamp(
      static_cast<int>(mean[0] + stddev[0] * config_.adapt_ratio_high),
      t_low + config_.min_t_gap, config_.max_t_high);
  
  if (config_.debug_mode) {
    std::cerr << "[DEBUG] Adaptive thresholds: mean=" << mean[0] 
              << ", std=" << stddev[0]
              << ", t_low=" << t_low 
              << ", t_high=" << t_high << std::endl;
  }
}

void ThreeFrameDiff::Reset() {
  prev2_.release();
  prev1_.release();
}

}  // namespace pellet::imgprocess
