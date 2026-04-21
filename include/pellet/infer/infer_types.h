#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace pellet::infer {

enum class ClassLabel : uint8_t {
  kBackground = 0,
  kPellet = 1,
};

struct InferenceRequest {
  uint32_t frame_id{0};
  std::vector<cv::Mat> rois;
};

struct InferenceResult {
  uint32_t frame_id{0};
  std::vector<float> scores;
};

}  // namespace pellet::infer
