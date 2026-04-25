#pragma once

#include <cstdint>

#include <opencv2/core/mat.hpp>

namespace pellet::detector {

struct FramePacket {
  uint32_t frame_id{0};
  int64_t timestamp_ms{0};
  cv::Mat frame_bgr;
};

}  // namespace pellet::detector

