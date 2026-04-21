#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "pellet/imgprocess/candidate_extractor.h"

namespace pellet::imgprocess {

struct RoiCropConfig {
  int output_size{32};
  float size_scale{2.2F};
  int min_crop{20};
  int max_crop{48};
};

struct RoiBatch {
  std::vector<cv::Mat> patches;
  std::vector<cv::Rect> boxes;
  std::vector<cv::Point2f> centers;
};

RoiBatch CropRoiBatch(
    const cv::Mat& gray_frame,
    const std::vector<Candidate>& candidates,
    const RoiCropConfig& config);

}  // namespace pellet::imgprocess
