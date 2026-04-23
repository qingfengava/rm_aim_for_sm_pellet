#pragma once

#include <cstdint>

#include <opencv2/core/types.hpp>

namespace pellet {

struct Detection {
  uint32_t frame_id{0};
  int64_t timestamp_ms{0};
  cv::Point2f center{0.0F, 0.0F};
  cv::Rect2f bbox{0.0F, 0.0F, 0.0F, 0.0F};
  float score{0.0F};
};

}  // namespace pellet
