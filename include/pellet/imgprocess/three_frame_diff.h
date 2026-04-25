#pragma once

#include <opencv2/core/mat.hpp>

namespace pellet::imgprocess {

struct ThreeFrameDiffConfig {
  // 自适应阈值参数
  float adapt_ratio_low{0.8f};   
  float adapt_ratio_high{2.5f};   
  
  // 阈值边界限制
  int min_t_low{10};
  int max_t_low{100};
  int min_t_high{20};
  int max_t_high{200};
  int min_t_gap{5};              
  
  // 静态保底阈值（当画面无运动时使用）
  int static_t_low{30};
  int static_t_high{50};
  
  bool debug_mode{false};
};

class ThreeFrameDiff {
public:
  explicit ThreeFrameDiff(const ThreeFrameDiffConfig& config = ThreeFrameDiffConfig());
 
  cv::Mat Apply(const cv::Mat& gray_frame);
  
  void Reset();

private:
  // 自适应计算双阈值（基于运动响应图）
  void ComputeAdaptiveThresholds(const cv::Mat& motion_response, int& t_low, int& t_high);
  
  ThreeFrameDiffConfig config_;
  cv::Mat prev1_;
  cv::Mat prev2_;
};

}  // namespace pellet::imgprocess
