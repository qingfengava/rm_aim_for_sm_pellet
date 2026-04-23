#include "pellet/infer/openvino_classifier.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if defined(PELLET_WITH_OPENVINO)
#include <openvino/openvino.hpp>
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

struct OpenVinoClassifier::Impl {
  bool initialized{false};
  bool runtime_available{false};
#if defined(PELLET_WITH_OPENVINO)
  std::shared_ptr<ov::Core> core;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;
  ov::Shape input_shape;
  bool nchw_input{true};
#endif
};

OpenVinoClassifier::OpenVinoClassifier() : impl_(std::make_unique<Impl>()) {}

OpenVinoClassifier::~OpenVinoClassifier() = default;

bool OpenVinoClassifier::Init(const InferenceConfig& config) {
  config_ = config;
  impl_->initialized = true;

#if defined(PELLET_WITH_OPENVINO)
  std::string model_path = config_.openvino_model_path;
  if (model_path.empty()) {
    model_path = config_.model_path;
  }
  if (!model_path.empty() && std::filesystem::exists(model_path)) {
    try {
      impl_->core = std::make_shared<ov::Core>();
      const std::string device = config_.device.empty() ? "CPU" : config_.device;
      impl_->compiled_model = impl_->core->compile_model(model_path, device);
      impl_->infer_request = impl_->compiled_model.create_infer_request();

      const ov::Output<const ov::Node> input_port = impl_->compiled_model.input();
      impl_->input_shape = input_port.get_shape();

      if (impl_->input_shape.size() == 4U) {
        impl_->nchw_input = (impl_->input_shape[1] == 1U);
        impl_->runtime_available = true;
      } else {
        impl_->runtime_available = false;
      }
    } catch (const std::exception&) {
      impl_->runtime_available = false;
    }
  }
#else
  impl_->runtime_available = false;
#endif

  return impl_->runtime_available;
}

std::vector<float> OpenVinoClassifier::Infer(const std::vector<cv::Mat>& rois) {
  if (!impl_->initialized || !impl_->runtime_available) {
    return std::vector<float>(rois.size(), 0.0F);
  }

  std::vector<float> scores;
  scores.reserve(rois.size());

#if defined(PELLET_WITH_OPENVINO)
  if (impl_->runtime_available) {
    const std::size_t h = impl_->nchw_input ? impl_->input_shape[2] : impl_->input_shape[1];
    const std::size_t w = impl_->nchw_input ? impl_->input_shape[3] : impl_->input_shape[2];

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

      cv::Mat resized;
      cv::resize(gray, resized, cv::Size{static_cast<int>(w), static_cast<int>(h)});

      cv::Mat input_f32;
      resized.convertTo(input_f32, CV_32F, 1.0 / 255.0);

      ov::Tensor input_tensor(ov::element::f32, impl_->input_shape);
      float* dst = input_tensor.data<float>();

      if (impl_->nchw_input) {
        for (std::size_t y = 0; y < h; ++y) {
          const float* row = input_f32.ptr<float>(static_cast<int>(y));
          std::copy(row, row + w, dst + y * w);
        }
      } else {
        for (std::size_t y = 0; y < h; ++y) {
          const float* row = input_f32.ptr<float>(static_cast<int>(y));
          for (std::size_t x = 0; x < w; ++x) {
            dst[(y * w + x)] = row[x];
          }
        }
      }

      try {
        impl_->infer_request.set_input_tensor(input_tensor);
        impl_->infer_request.infer();
        const ov::Tensor output_tensor = impl_->infer_request.get_output_tensor();
        const float* out = output_tensor.data<const float>();
        const std::size_t out_size = output_tensor.get_size();
        float score = (out_size >= 2U) ? out[1] : out[0];
        score = std::clamp(score, 0.0F, 1.0F);
        scores.push_back(score);
      } catch (const std::exception&) {
        scores.push_back(FallbackScore(roi));
      }
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
