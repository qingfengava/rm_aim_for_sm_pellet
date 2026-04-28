#pragma once

#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "pellet/config.hpp"
#include "pellet/infer/i_classifier.hpp"

namespace pellet::infer {

class OnnxClassifier final : public IClassifier {
 public:
  OnnxClassifier();
  ~OnnxClassifier() override;

  bool Init(const InferenceConfig& config, const InferRuntimeOptions& runtime_options) override;
  std::vector<float> Infer(const std::vector<cv::Mat>& rois) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  InferenceConfig config_{};
};

}  // namespace pellet::infer
