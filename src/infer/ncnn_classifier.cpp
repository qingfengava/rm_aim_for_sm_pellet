#include "pellet/infer/ncnn_classifier.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if defined(PELLET_WITH_NCNN)
#include <ncnn/net.h>
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

struct NcnnClassifier::Impl {
  bool initialized{false};
  bool runtime_available{false};
#if defined(PELLET_WITH_NCNN)
  ncnn::Net net;
#endif
};

NcnnClassifier::NcnnClassifier() : impl_(std::make_unique<Impl>()) {}

NcnnClassifier::~NcnnClassifier() = default;

bool NcnnClassifier::Init(const InferenceConfig& config) {
  config_ = config;
  impl_->initialized = true;
  impl_->runtime_available = false;

#if defined(PELLET_WITH_NCNN)
  std::string param_path = config_.ncnn_param_path;
  std::string bin_path = config_.ncnn_bin_path;

  if (param_path.empty()) {
    param_path = config_.model_path;
  }
  if (bin_path.empty() && param_path.size() > 6U && param_path.substr(param_path.size() - 6U) == ".param") {
    bin_path = param_path.substr(0, param_path.size() - 6U) + ".bin";
  }

  if (!param_path.empty() && !bin_path.empty() && std::filesystem::exists(param_path) &&
      std::filesystem::exists(bin_path)) {
    impl_->net.opt.num_threads = std::max(1, config_.num_threads);
    impl_->net.opt.use_vulkan_compute = false;
    impl_->net.opt.use_fp16_storage = config_.use_fp16;
    impl_->net.opt.use_fp16_arithmetic = config_.use_fp16;

    const int load_param_ok = impl_->net.load_param(param_path.c_str());
    const int load_model_ok = impl_->net.load_model(bin_path.c_str());
    impl_->runtime_available = (load_param_ok == 0 && load_model_ok == 0);
  }
#endif

  return impl_->runtime_available;
}

std::vector<float> NcnnClassifier::Infer(const std::vector<cv::Mat>& rois) {
  if (!impl_->initialized || !impl_->runtime_available) {
    return std::vector<float>(rois.size(), 0.0F);
  }

  std::vector<float> scores;
  scores.reserve(rois.size());

#if defined(PELLET_WITH_NCNN)
  if (impl_->runtime_available) {
    for (const auto& roi : rois) {
      if (roi.empty()) {
        scores.push_back(0.0F);
        continue;
      }

      cv::Mat gray;
      if (roi.channels() == 1) {
        gray = roi;
      } else {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
      }

      ncnn::Mat input = ncnn::Mat::from_pixels(gray.data, ncnn::Mat::PIXEL_GRAY, gray.cols, gray.rows);
      const float norm[1] = {1.0F / 255.0F};
      input.substract_mean_normalize(nullptr, norm);

      ncnn::Extractor ex = impl_->net.create_extractor();

      if (ex.input(config_.input_blob_name.c_str(), input) != 0) {
        scores.push_back(FallbackScore(roi));
        continue;
      }

      ncnn::Mat out;
      if (ex.extract(config_.output_blob_name.c_str(), out) != 0 || out.empty()) {
        scores.push_back(FallbackScore(roi));
        continue;
      }

      float score = out[0];
      if (out.total() >= 2U) {
        score = out[1];
      }
      scores.push_back(std::clamp(score, 0.0F, 1.0F));
    }
    return scores;
  }
#endif

  for (const auto& roi : rois) {
    scores.push_back(FallbackScore(roi));
  }

  return scores;
}

}  // namespace pellet::infer
