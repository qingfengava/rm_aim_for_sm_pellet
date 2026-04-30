#pragma once

#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

namespace pellet::imgprocess {

struct BgSubtractConfig {
  std::string backend{"knn"};  // "knn" | "mog2"
  int history{30};
  double var_threshold{14.0};
  double learning_rate{-1.0};
  bool detect_shadows{false};
  int downsample{2};  // 1=原生分辨率; 2=1/2; ...
};

struct BgSubtractResult {
  cv::Mat binary_mask;       // 二值前景 (0/255)，已恢复到原始分辨率
  cv::Mat motion_response;   // 单帧差 |prev - curr| (0-255)，原始分辨率
};

class BgSubtractor {
 public:
  explicit BgSubtractor(const BgSubtractConfig& config = BgSubtractConfig());
  ~BgSubtractor();

  BgSubtractResult Apply(const cv::Mat& gray_frame);

  void Reset();

  void SetLearningRate(double rate);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pellet::imgprocess
