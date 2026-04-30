#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "pellet/imgprocess/candidate_extractor.hpp"

namespace pellet::imgprocess {

struct RoiCropConfig {
  int output_size{32};
  float size_scale{2.2F};
  int min_crop{20};
  int max_crop{48};
};

struct RoiCropStats {
  int total_candidates{0};
  int valid_crops{0};
  int filtered_low_quality{0};
  int filtered_low_texture{0};
  int filtered_out_of_bounds{0};
  float avg_crop_size{0.0f};
};

struct RoiBatch {
  std::vector<cv::Mat> patches;
  std::vector<cv::Rect> boxes;
  std::vector<cv::Point2f> centers;
};

RoiBatch CropRoiBatch(
    const cv::Mat& gray_frame,
    const std::vector<Candidate>& candidates,
    const RoiCropConfig& config,
    RoiCropStats* stats = nullptr);

cv::Mat CropSingleRoi(
    const cv::Mat& gray_frame,
    const cv::Rect& roi,
    int output_size = 32,
    bool debug_mode = false);

}  // namespace pellet::imgprocess
