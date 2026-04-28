#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "pellet/config.hpp"

namespace pellet::infer {

struct InferRuntimeOptions {
  bool debug_log_init{false};
};

class IClassifier {
 public:
  virtual ~IClassifier() = default;

  virtual bool Init(
      const InferenceConfig& config,
      const InferRuntimeOptions& runtime_options) = 0;
  virtual std::vector<float> Infer(const std::vector<cv::Mat>& rois) = 0;
};

std::shared_ptr<IClassifier> CreateClassifier(const std::string& backend);

}  // namespace pellet::infer
