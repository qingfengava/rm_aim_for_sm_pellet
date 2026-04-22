#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace pellet::imgprocess {

struct Candidate {
  cv::Rect bbox;
  cv::Point2f center;
  int area{0};
  float motion_score{0.0F};
  float brightness{0.0F};
  float circularity{0.0F};
  float aspect_ratio{1.0F};
  float extent{0.0F};
  float local_contrast{0.0F};
  float rank_score{0.0F};
};

std::vector<Candidate> ExtractCandidates(
    const cv::Mat& binary_mask,
    const cv::Mat& gray_frame,
    const cv::Mat& motion_response);

}  // namespace pellet::imgprocess
