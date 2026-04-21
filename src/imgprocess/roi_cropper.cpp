#include "pellet/imgprocess/roi_cropper.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "pellet/utils/image_utils.h"

namespace pellet::imgprocess {

RoiBatch CropRoiBatch(
    const cv::Mat& gray_frame,
    const std::vector<Candidate>& candidates,
    const RoiCropConfig& config) {
  RoiBatch batch;
  if (gray_frame.empty() || candidates.empty()) {
    return batch;
  }

  batch.patches.reserve(candidates.size());
  batch.boxes.reserve(candidates.size());
  batch.centers.reserve(candidates.size());

  for (const auto& candidate : candidates) {
    const float side_f = std::sqrt(static_cast<float>(std::max(1, candidate.area))) * config.size_scale;
    const int side = std::clamp(
        static_cast<int>(std::round(side_f)),
        std::max(1, config.min_crop),
        std::max(config.min_crop, config.max_crop));

    const cv::Rect roi = utils::MakeSquareRoi(candidate.center, side, gray_frame.size());
    if (roi.width <= 0 || roi.height <= 0) {
      continue;
    }

    cv::Mat resized;
    cv::resize(
        gray_frame(roi),
        resized,
        cv::Size{std::max(1, config.output_size), std::max(1, config.output_size)},
        0.0,
        0.0,
        cv::INTER_LINEAR);

    batch.patches.push_back(resized);
    batch.boxes.push_back(roi);
    batch.centers.push_back(candidate.center);
  }

  return batch;
}

}  // namespace pellet::imgprocess
