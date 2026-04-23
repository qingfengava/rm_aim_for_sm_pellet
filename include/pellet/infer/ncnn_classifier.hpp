#pragma once

#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "pellet/config.hpp"
#include "pellet/infer/i_classifier.hpp"

namespace pellet::infer {

class NcnnClassifier final : public IClassifier {
 public:
  NcnnClassifier();
  ~NcnnClassifier() override;

  bool Init(const InferenceConfig& config) override;
  std::vector<float> Infer(const std::vector<cv::Mat>& rois) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  InferenceConfig config_{};
};

}  // namespace pellet::infer
