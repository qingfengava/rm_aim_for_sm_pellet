#include "pellet/imgprocess/motion_confirm.hpp"

#include <cstdint>
#include <opencv2/core.hpp>

namespace pellet::imgprocess {

namespace {

constexpr int kDirectScanAreaLimit = 50;

double ComputeMeanSmallRoi(const cv::Mat& src, const cv::Rect& roi) {
  int sum = 0;
  for (int y = roi.y; y < roi.y + roi.height; ++y) {
    const std::uint8_t* row = src.ptr<std::uint8_t>(y);
    for (int x = roi.x; x < roi.x + roi.width; ++x) {
      sum += row[x];
    }
  }
  return static_cast<double>(sum) / static_cast<double>(roi.area());
}

}  // namespace

std::vector<Candidate> FilterByMotionConfirm(
    const std::vector<Candidate>& candidates,
    const cv::Mat& motion_response,
    int threshold) {
  if (motion_response.empty()) {
    return candidates;
  }

  std::vector<Candidate> confirmed;
  confirmed.reserve(candidates.size());

  const cv::Rect frame_rect{0, 0, motion_response.cols, motion_response.rows};

  for (const auto& c : candidates) {
    const cv::Rect roi = c.bbox & frame_rect;
    const int area = roi.area();
    if (area <= 0) {
      continue;
    }

    const double mean_motion = (area < kDirectScanAreaLimit)
        ? ComputeMeanSmallRoi(motion_response, roi)
        : cv::mean(motion_response(roi))[0];

    if (mean_motion >= static_cast<double>(threshold)) {
      confirmed.push_back(c);
    }
  }

  return confirmed;
}

}  // namespace pellet::imgprocess
