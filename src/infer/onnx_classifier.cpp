#include "pellet/infer/onnx_classifier.h"

#include <vector>

namespace pellet::infer {

struct OnnxClassifier::Impl {
  bool initialized{false};
};

OnnxClassifier::OnnxClassifier() : impl_(std::make_unique<Impl>()) {}

OnnxClassifier::~OnnxClassifier() = default;

bool OnnxClassifier::Init(const InferenceConfig& config) {
  config_ = config;
  impl_->initialized = true;
  return true;
}

std::vector<float> OnnxClassifier::Infer(const std::vector<cv::Mat>& rois) {
  std::vector<float> scores(rois.size(), 0.5F);
  if (!impl_->initialized) {
    return std::vector<float>(rois.size(), 0.0F);
  }
  return scores;
}

}  // namespace pellet::infer
