#include "pellet/infer/i_classifier.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "pellet/infer/ncnn_classifier.h"
#include "pellet/infer/onnx_classifier.h"
#include "pellet/infer/openvino_classifier.h"
#include "pellet/infer/tensorrt_classifier.h"

namespace pellet::infer {
namespace {

class MockClassifier final : public IClassifier {
 public:
  bool Init(const InferenceConfig& config) override {
    config_ = config;
    return true;
  }

  std::vector<float> Infer(const std::vector<cv::Mat>& rois) override {
    std::vector<float> scores;
    scores.reserve(rois.size());

    for (const auto& roi : rois) {
      if (roi.empty()) {
        scores.push_back(0.0F);
        continue;
      }
      const cv::Scalar mean_val = cv::mean(roi);
      scores.push_back(static_cast<float>(mean_val[0] / 255.0));
    }

    return scores;
  }

 private:
  InferenceConfig config_{};
};

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

}  // namespace

std::shared_ptr<IClassifier> CreateClassifier(const std::string& backend) {
  const std::string normalized = ToLower(backend);
  if (normalized == "onnx" || normalized == "onnxruntime" || normalized == "ort") {
    return std::make_shared<OnnxClassifier>();
  }
  if (normalized == "tensorrt" || normalized == "trt") {
    return std::make_shared<TensorRtClassifier>();
  }
  if (normalized == "openvino" || normalized == "ov") {
    return std::make_shared<OpenVinoClassifier>();
  }
  if (normalized == "ncnn") {
    return std::make_shared<NcnnClassifier>();
  }
  return std::make_shared<MockClassifier>();
}

}  // namespace pellet::infer
