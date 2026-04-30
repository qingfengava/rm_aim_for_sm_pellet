#pragma once

#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "pellet/config.hpp"
#include "pellet/infer/i_classifier.hpp"

namespace pellet::infer {

class TensorRtClassifier final : public IClassifier {
 public:
  TensorRtClassifier();
  ~TensorRtClassifier() override;

  bool Init(const InferenceConfig& config, const InferRuntimeOptions& runtime_options) override;
  std::vector<float> Infer(const std::vector<cv::Mat>& rois) override;
  InferRuntimeState GetRuntimeState() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  InferenceConfig config_{};
};

}  // namespace pellet::infer
