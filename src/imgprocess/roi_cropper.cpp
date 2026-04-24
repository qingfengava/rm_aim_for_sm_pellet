#include "pellet/imgprocess/roi_cropper.h"

#include <algorithm>
#include <cmath>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include "pellet/utils/image_utils.h"

namespace pellet::imgprocess {

RoiBatch CropRoiBatch(
    const cv::Mat& gray_frame,
    const std::vector<Candidate>& candidates,
    const RoiCropConfig& config,
    RoiCropStats* stats) {

  RoiBatch batch;
  RoiCropStats local_stats;

  //输入检查
  if (gray_frame.empty() || candidates.empty()) {
    if (config.debug_mode) {
      std::cerr<<"[WARNING] CropRoiBatch:empty input\n";
    }
    return batch;
  }

  batch.patches.reserve(candidates.size());
  batch.boxes.reserve(candidates.size());
  batch.centers.reserve(candidates.size());

  for (const auto& candidate : candidates) {
    local_stats.total_candidates++;

    //边界检查
    if (candidate.center.x<0||candidate.center.x>=gray_frame.cols||
        candidate.center.y<0||candidate.center.y>=gray_frame.rows) {
          local_stats.filtered_out_of_bounds++;
          if (config.debug_mode) {
            std::cerr<<"[WARNING] Candidates out of bounds:"<<candidate.center<<"\n";
          }
          continue;
    }

    //计算尺寸
    const float side_f = std::sqrt(static_cast<float>(std::max(1, candidate.area))) * config.size_scale;
    const int side = std::clamp(
        static_cast<int>(std::round(side_f)),
        std::max(1, config.min_crop),
        std::max(config.min_crop, config.max_crop));

    //生成roi
    const cv::Rect roi = utils::MakeSquareRoi(candidate.center, side, gray_frame.size());
    if (roi.width <= 0 || roi.height <= 0) {
      local_stats.filtered_out_of_bounds++;
      continue;
    }

    //质量预过滤
    if (config.enable_quality_filter) {
      cv::Mat roi_patch=gray_frame(roi);

      //方差检查(背景)
      cv::Scalar mean,stddev;
      cv::meanStdDev(roi_patch, mean, stddev);
      if (stddev[0]<config.min_variance) {
        local_stats.filtered_low_variance++;
        if (config.debug_mode) {
          std::cerr<<"[DEBUG] Low variance patch filtered:var="<<stddev[0]<<"\n";
        }
        continue;
      }
    }

    //缩放
    cv::Mat resized;
    cv::resize(
        gray_frame(roi),
        resized,
        cv::Size{std::max(1, config.output_size), std::max(1, config.output_size)},
        0.0,
        0.0,
        cv::INTER_LINEAR);

    if (resized.empty()||resized.total()==0) {
      if (config.debug_mode) {
        std::cerr<<"[WARNING] Resize failed fro ROI:"<<roi<<"\n";
      }
      continue;
    }

    batch.patches.push_back(resized);
    batch.boxes.push_back(roi);
    batch.centers.push_back(candidate.center);
    local_stats.valid_crops++;
    local_stats.avg_crop_size+=side;
  }

  //统计输出
  if (stats) {
        *stats = local_stats;
    }
    
    if (config.debug_mode && local_stats.total_candidates > 0) {
        std::cerr << "[INFO] CropRoiBatch: total=" << local_stats.total_candidates
                  << ", valid=" << local_stats.valid_crops
                  << ", out_of_bounds=" << local_stats.filtered_out_of_bounds
                  << ", low_var=" << local_stats.filtered_low_variance
                  << ", low_contrast=" << local_stats.filtered_low_contrast << "\n";
    }

  return batch;
}

}  // namespace pellet::imgprocess
