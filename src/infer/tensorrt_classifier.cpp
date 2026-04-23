#include "pellet/infer/tensorrt_classifier.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>

#if defined(PELLET_WITH_TENSORRT)
#include <NvInfer.h>
#endif

namespace pellet::infer {
namespace {

float FallbackScore(const cv::Mat& roi) {
  if (roi.empty()) {
    return 0.0F;
  }
  const cv::Scalar mean_val = cv::mean(roi);
  const float score = static_cast<float>(mean_val[0] / 255.0);
  return std::clamp(score, 0.0F, 1.0F);
}

}  // namespace

struct TensorRtClassifier::Impl {
  bool initialized{false};
  bool runtime_available{false};
};

TensorRtClassifier::TensorRtClassifier() : impl_(std::make_unique<Impl>()) {}

TensorRtClassifier::~TensorRtClassifier() = default;

bool TensorRtClassifier::Init(const InferenceConfig& config) {
  config_ = config;
  impl_->initialized = true;

#if defined(PELLET_WITH_TENSORRT)
  std::string engine_path = config_.trt_engine_path;
  if (engine_path.empty()) {
    engine_path = config_.model_path;
  }
  impl_->runtime_available = !engine_path.empty() && std::filesystem::exists(engine_path);
#else
  impl_->runtime_available = false;
#endif

  return impl_->runtime_available;
}

std::vector<float> TensorRtClassifier::Infer(const std::vector<cv::Mat>& rois) {
  if (!impl_->initialized || !impl_->runtime_available) {
    return std::vector<float>(rois.size(), 0.0F);
  }

  std::vector<float> scores;
  scores.reserve(rois.size());

  for (const auto& roi : rois) {
    scores.push_back(FallbackScore(roi));
  }

  return scores;
}

}  // namespace pellet::infer
