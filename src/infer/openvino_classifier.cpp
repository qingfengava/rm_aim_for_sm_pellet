#include "pellet/infer/openvino_classifier.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <exception>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <wust_vl/common/utils/logger.hpp>

#include "pellet/utils/infer_utils.hpp"
#include "pellet/utils/debug_utils.hpp"

#if defined(PELLET_WITH_OPENVINO)
#include <wust_vl/ml_net/openvino/openvino_net.hpp>
#endif

namespace pellet::infer {
namespace {

constexpr const char* kBackendName = "openvino";
constexpr const char* kWarnKeyCreateRequestFailed = "openvino_create_request_failed";
constexpr const char* kWarnKeyValidMaskNotReady = "openvino_valid_mask_not_ready";
constexpr const char* kWarnKeyRequestRetry = "openvino_request_retry";
constexpr const char* kWarnKeyInferFail = "openvino_infer_fail";
constexpr const char* kWarnKeyCooldown = "openvino_cooldown";
constexpr int kInferFailureCooldownThreshold = 3;
constexpr auto kInferFailureCooldown = std::chrono::milliseconds(800);

void LogInitDebug(
    const InferenceConfig& config,
    const InferRuntimeOptions& runtime_options,
    const std::string& model_resolved_path,
    const char* actual_mode) {
  if (!runtime_options.debug_log_init) {
    return;
  }
  WUST_INFO("infer") << "backend=" << kBackendName
                     << ", precision=" << InferToLower(config.precision)
                     << ", batch_size=" << InferResolveBatchSize(config.batch_size)
                     << ", model_resolved_path=" << model_resolved_path
                     << ", actual_mode=" << actual_mode;
}

std::string ResolveModelPath(const InferenceConfig& config) {
  const std::string precision = InferToLower(config.precision);
  std::vector<std::string> candidates;
  if (precision == "int8") {
    if (!config.openvino_model_path.empty()) {
      candidates.push_back(config.openvino_model_path);
    }
    if (!config.int8_model_path.empty()) {
      candidates.push_back(config.int8_model_path);
    }
    if (!config.model_path.empty()) {
      candidates.push_back(config.model_path);
    }
  } else {
    if (!config.openvino_model_path.empty()) {
      candidates.push_back(config.openvino_model_path);
    }
    if (!config.model_path.empty()) {
      candidates.push_back(config.model_path);
    }
  }

  for (const auto& path : candidates) {
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  return candidates.empty() ? std::string() : candidates.front();
}

#if defined(PELLET_WITH_OPENVINO)
float ExtractScoreOv(const ov::Tensor& output) {
  if (output.get_size() == 0) {
    return 0.0F;
  }
  if (output.get_element_type() != ov::element::f32) {
    return 0.0F;
  }
  const float* out = output.data<const float>();
  const float score = output.get_size() >= 2U ? out[1] : out[0];
  return std::clamp(score, 0.0F, 1.0F);
}

float ExtractBatchScoreAtOv(const ov::Tensor& output, std::size_t batch_size, std::size_t batch_index) {
  batch_size = std::max<std::size_t>(1, batch_size);
  if (output.get_size() == 0 || output.get_element_type() != ov::element::f32) {
    return 0.0F;
  }

  const float* out = output.data<const float>();
  const std::size_t total = output.get_size();
  if (batch_size == 1U) {
    return batch_index == 0U ? ExtractScoreOv(output) : 0.0F;
  }
  if (total < batch_size) {
    return ExtractScoreOv(output);
  }
  const std::size_t stride = std::max<std::size_t>(1, total / batch_size);
  const std::size_t start = batch_index * stride;
  if (start >= total) {
    return 0.0F;
  }
  const std::size_t remaining = total - start;
  const std::size_t sample_count = std::min(stride, remaining);
  const float score = sample_count >= 2U ? out[start + 1U] : out[start];
  return std::clamp(score, 0.0F, 1.0F);
}
#endif

}  // namespace

struct OpenVinoClassifier::Impl {
  bool initialized{false};
  bool runtime_available{false};
  bool debug_log_infer{false};
  int consecutive_failures{0};
  std::chrono::steady_clock::time_point cooldown_until{};
  bool degraded_last_call{false};
  bool batch_disabled{false};
#if defined(PELLET_WITH_OPENVINO)
  std::unique_ptr<wust_vl::ml_net::OpenvinoNet> net;
  std::unique_ptr<ov::InferRequest> infer_request;
  ov::Shape input_shape;
  int batch_size{1};
  ov::Tensor input_tensor_scratch;
  bool input_tensor_scratch_ready{false};
  std::vector<uint8_t> valid_mask_scratch;
#endif
};

OpenVinoClassifier::OpenVinoClassifier() : impl_(std::make_unique<Impl>()) {}

OpenVinoClassifier::~OpenVinoClassifier() = default;

bool OpenVinoClassifier::Init(
    const InferenceConfig& config,
    const InferRuntimeOptions& runtime_options) {
  config_ = config;
  impl_->initialized = true;
  impl_->runtime_available = false;
  impl_->debug_log_infer = runtime_options.debug_log_init;
  impl_->consecutive_failures = 0;
  impl_->cooldown_until = std::chrono::steady_clock::time_point{};
  impl_->degraded_last_call = false;
  impl_->batch_disabled = false;
#if defined(PELLET_WITH_OPENVINO)
  impl_->input_tensor_scratch_ready = false;
  impl_->valid_mask_scratch.clear();
  impl_->infer_request.reset();
#endif

#if defined(PELLET_WITH_OPENVINO)
  const std::string precision = InferToLower(config_.precision);
  if (!InferIsSupportedPrecision(precision)) {
    WUST_ERROR("openvino_classifier")
        << "unsupported precision for OpenVINO backend: " << config_.precision;
    return false;
  }

  const std::string model_path = ResolveModelPath(config_);
  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    WUST_ERROR("openvino_classifier")
        << "model path not found, model_resolved_path=" << model_path;
    return false;
  }

  try {
    wust_vl::ml_net::OpenvinoNet::Params params;
    params.model_path = model_path;
    params.device_name = config_.device.empty() ? "CPU" : config_.device;
    params.mode = ov::hint::PerformanceMode::LATENCY;

    impl_->net = std::make_unique<wust_vl::ml_net::OpenvinoNet>();
    impl_->runtime_available = impl_->net->init(
        params, [](ov::preprocess::PrePostProcessor&) {});
    if (!impl_->runtime_available) {
      impl_->net.reset();
      return false;
    }

    auto [input_type, input_shape] = impl_->net->getInputInfo();
    if (input_type != ov::element::f32 || input_shape.size() != 4U) {
      WUST_ERROR("openvino_classifier")
          << "unsupported OpenVINO input tensor shape/type";
      impl_->runtime_available = false;
      impl_->net.reset();
      return false;
    }
    impl_->input_shape = input_shape;
    if (static_cast<int>(input_shape[3]) <= 0 || static_cast<int>(input_shape[2]) <= 0) {
      WUST_ERROR("openvino_classifier")
          << "invalid OpenVINO input spatial size";
      impl_->runtime_available = false;
      impl_->net.reset();
      return false;
    }
    impl_->batch_size = InferResolveBatchSize(config_.batch_size);
    impl_->input_tensor_scratch = ov::Tensor(ov::element::f32, impl_->input_shape);
    impl_->input_tensor_scratch_ready = true;
    impl_->infer_request = std::make_unique<ov::InferRequest>(impl_->net->createInferRequest());
    impl_->valid_mask_scratch.assign(
        std::max<std::size_t>(
            std::max<std::size_t>(1U, impl_->input_shape.empty() ? 1U : impl_->input_shape[0]),
            static_cast<std::size_t>(InferResolveBatchSize(config_.batch_size))),
        static_cast<uint8_t>(0));
    const bool true_batch_mode = impl_->input_shape[0] > 1U;
    LogInitDebug(
        config_,
        runtime_options,
        model_path,
        true_batch_mode ? "true-batch" : "micro-batch");
  } catch (const std::exception&) {
    WUST_ERROR("openvino_classifier") << "OpenVINO backend init failed";
    impl_->runtime_available = false;
    impl_->input_tensor_scratch_ready = false;
    impl_->infer_request.reset();
    impl_->net.reset();
  }
#endif

  return impl_->runtime_available;
}

std::vector<float> OpenVinoClassifier::Infer(const std::vector<cv::Mat>& rois) {
  impl_->degraded_last_call = false;
  if (!impl_->initialized || !impl_->runtime_available) {
    impl_->degraded_last_call = true;
    return std::vector<float>(rois.size(), 0.0F);
  }

  std::vector<float> scores(rois.size(), 0.0F);

#if defined(PELLET_WITH_OPENVINO)
  if (impl_->net == nullptr || !impl_->input_tensor_scratch_ready) {
    impl_->degraded_last_call = true;
    return scores;
  }
  const auto now = std::chrono::steady_clock::now();
  if (impl_->cooldown_until > now) {
    impl_->degraded_last_call = true;
    if (utils::ShouldLogRateLimited("infer", kWarnKeyCooldown)) {
      const auto cooldown_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   impl_->cooldown_until - now)
                                   .count();
      WUST_WARN("infer")
          << "backend=" << kBackendName
          << ", infer temporarily disabled due to recent failures, cooldown_ms="
          << cooldown_ms;
    }
    return scores;
  }
  if (impl_->infer_request == nullptr) {
    try {
      impl_->infer_request = std::make_unique<ov::InferRequest>(impl_->net->createInferRequest());
    } catch (const std::exception& e) {
      impl_->degraded_last_call = true;
      ++impl_->consecutive_failures;
      if (impl_->consecutive_failures >= kInferFailureCooldownThreshold) {
        impl_->cooldown_until = std::chrono::steady_clock::now() + kInferFailureCooldown;
      }
      if (utils::ShouldLogRateLimited("infer", kWarnKeyCreateRequestFailed)) {
        if (impl_->debug_log_infer) {
          WUST_WARN("infer")
              << "backend=" << kBackendName
              << ", create infer request failed"
              << ", consecutive_failures=" << impl_->consecutive_failures
              << ", cooldown_active="
              << (impl_->cooldown_until > std::chrono::steady_clock::now() ? 1 : 0)
              << ", err=" << e.what();
        } else {
          WUST_WARN("infer")
              << "backend=" << kBackendName
              << ", create infer request failed"
              << ", consecutive_failures=" << impl_->consecutive_failures
              << ", cooldown_active="
              << (impl_->cooldown_until > std::chrono::steady_clock::now() ? 1 : 0);
        }
      }
      return scores;
    } catch (...) {
      impl_->degraded_last_call = true;
      ++impl_->consecutive_failures;
      if (impl_->consecutive_failures >= kInferFailureCooldownThreshold) {
        impl_->cooldown_until = std::chrono::steady_clock::now() + kInferFailureCooldown;
      }
      if (utils::ShouldLogRateLimited("infer", kWarnKeyCreateRequestFailed)) {
        WUST_WARN("infer")
            << "backend=" << kBackendName
            << ", create infer request failed"
            << ", consecutive_failures=" << impl_->consecutive_failures
            << ", cooldown_active="
            << (impl_->cooldown_until > std::chrono::steady_clock::now() ? 1 : 0);
      }
      return scores;
    }
  }
  float* tensor_ptr = impl_->input_tensor_scratch.data<float>();
  const std::size_t tensor_size = impl_->input_tensor_scratch.get_size();
  if (tensor_ptr == nullptr || tensor_size == 0U) {
    return scores;
  }

  const std::size_t requested_chunk_size =
      static_cast<std::size_t>(InferResolveBatchSize(impl_->batch_size));
  const std::size_t model_batch_size =
      (impl_->input_shape.empty() ? 1U : std::max<std::size_t>(1U, impl_->input_shape[0]));
  const bool model_supports_true_batch = model_batch_size > 1U && !impl_->batch_disabled;
  const std::size_t chunk_size =
      model_supports_true_batch ? model_batch_size
                                : requested_chunk_size;
  if (model_supports_true_batch && impl_->valid_mask_scratch.size() < chunk_size) {
    impl_->degraded_last_call = true;
    if (utils::ShouldLogRateLimited("infer", kWarnKeyValidMaskNotReady)) {
      WUST_WARN("infer")
          << "backend=" << kBackendName
          << ", valid_mask_scratch not ready, valid_mask=" << impl_->valid_mask_scratch.size()
          << ", expected_mask=" << chunk_size;
    }
    return scores;
  }
  bool exception_from_batch_path = false;
  auto infer_with_request_retry = [this](const ov::Tensor& input_tensor) -> ov::Tensor {
    if (impl_->infer_request == nullptr) {
      impl_->infer_request = std::make_unique<ov::InferRequest>(impl_->net->createInferRequest());
    }
    try {
      return impl_->net->infer(input_tensor, *impl_->infer_request);
    } catch (const std::exception& e) {
      if (utils::ShouldLogRateLimited("infer", kWarnKeyRequestRetry)) {
        if (impl_->debug_log_infer) {
          WUST_WARN("infer")
              << "backend=" << kBackendName
              << ", infer request failed once, recreate and retry, err=" << e.what();
        } else {
          WUST_WARN("infer")
              << "backend=" << kBackendName
              << ", infer request failed once, recreate and retry";
        }
      }
      impl_->infer_request.reset();
      impl_->infer_request = std::make_unique<ov::InferRequest>(impl_->net->createInferRequest());
      return impl_->net->infer(input_tensor, *impl_->infer_request);
    } catch (...) {
      if (utils::ShouldLogRateLimited("infer", kWarnKeyRequestRetry)) {
        WUST_WARN("infer")
            << "backend=" << kBackendName
            << ", infer request failed once, recreate and retry";
      }
      impl_->infer_request.reset();
      impl_->infer_request = std::make_unique<ov::InferRequest>(impl_->net->createInferRequest());
      return impl_->net->infer(input_tensor, *impl_->infer_request);
    }
  };
  try {
    for (std::size_t chunk_begin = 0; chunk_begin < rois.size(); chunk_begin += chunk_size) {
      const std::size_t chunk_count = std::min(chunk_size, rois.size() - chunk_begin);

      if (model_supports_true_batch && chunk_count > 0U && impl_->input_shape.size() >= 1U) {
        exception_from_batch_path = true;
        const std::size_t run_batch = model_batch_size;
        std::fill_n(tensor_ptr, tensor_size, 0.0F);

        const std::size_t sample_size = run_batch > 0U ? (tensor_size / run_batch) : 0U;
        std::fill_n(
            impl_->valid_mask_scratch.data(),
            chunk_count,
            static_cast<uint8_t>(0));
        std::size_t valid_count = 0U;
        for (std::size_t i = 0; i < chunk_count; ++i) {
          if (sample_size == 0U) {
            continue;
          }
          const bool filled = FillInputTensor(
              rois[chunk_begin + i],
              tensor_ptr + static_cast<std::ptrdiff_t>(i * sample_size),
              sample_size);
          if (!filled) {
            continue;
          }
          impl_->valid_mask_scratch[i] = 1U;
          ++valid_count;
        }
        if (valid_count == 0U) {
          exception_from_batch_path = false;
          continue;
        }

        const ov::Tensor output_tensor = infer_with_request_retry(impl_->input_tensor_scratch);
        for (std::size_t i = 0; i < chunk_count; ++i) {
          if (impl_->valid_mask_scratch[i] == 0U) {
            continue;
          }
          scores[chunk_begin + i] = ExtractBatchScoreAtOv(output_tensor, run_batch, i);
        }
        exception_from_batch_path = false;
        continue;
      }

      exception_from_batch_path = false;
      for (std::size_t i = 0; i < chunk_count; ++i) {
        std::fill_n(tensor_ptr, tensor_size, 0.0F);
        const bool filled = FillInputTensor(
            rois[chunk_begin + i],
            tensor_ptr,
            tensor_size);
        if (!filled) {
          continue;
        }

        const ov::Tensor output_tensor = infer_with_request_retry(impl_->input_tensor_scratch);
        scores[chunk_begin + i] = ExtractScoreOv(output_tensor);
      }
    }
    impl_->consecutive_failures = 0;
    impl_->cooldown_until = std::chrono::steady_clock::time_point{};
  } catch (const std::exception& e) {
    impl_->degraded_last_call = true;
    ++impl_->consecutive_failures;
    if (impl_->consecutive_failures >= kInferFailureCooldownThreshold) {
      impl_->cooldown_until = std::chrono::steady_clock::now() + kInferFailureCooldown;
    }
    if (exception_from_batch_path) {
      impl_->batch_disabled = true;
    }
    if (utils::ShouldLogRateLimited("infer", kWarnKeyInferFail)) {
      if (impl_->debug_log_infer) {
        WUST_WARN("infer")
            << "backend=" << kBackendName
            << ", infer failed once, fallback remaining samples to 0.0"
            << ", batch_disabled=" << (impl_->batch_disabled ? 1 : 0)
            << ", consecutive_failures=" << impl_->consecutive_failures
            << ", cooldown_active="
            << (impl_->cooldown_until > std::chrono::steady_clock::now() ? 1 : 0)
            << ", err=" << e.what();
      } else {
        WUST_WARN("infer")
            << "backend=" << kBackendName
            << ", infer failed once, fallback remaining samples to 0.0"
            << ", batch_disabled=" << (impl_->batch_disabled ? 1 : 0)
            << ", consecutive_failures=" << impl_->consecutive_failures
            << ", cooldown_active="
            << (impl_->cooldown_until > std::chrono::steady_clock::now() ? 1 : 0);
      }
    }
    return scores;
  } catch (...) {
    impl_->degraded_last_call = true;
    ++impl_->consecutive_failures;
    if (impl_->consecutive_failures >= kInferFailureCooldownThreshold) {
      impl_->cooldown_until = std::chrono::steady_clock::now() + kInferFailureCooldown;
    }
    if (exception_from_batch_path) {
      impl_->batch_disabled = true;
    }
    if (utils::ShouldLogRateLimited("infer", kWarnKeyInferFail)) {
      WUST_WARN("infer")
          << "backend=" << kBackendName
          << ", infer failed once, fallback remaining samples to 0.0"
          << ", batch_disabled=" << (impl_->batch_disabled ? 1 : 0)
          << ", consecutive_failures=" << impl_->consecutive_failures
          << ", cooldown_active="
          << (impl_->cooldown_until > std::chrono::steady_clock::now() ? 1 : 0);
    }
    return scores;
  }
  return scores;
#endif

  return scores;
}

InferRuntimeState OpenVinoClassifier::GetRuntimeState() const {
  InferRuntimeState state;
  state.consecutive_failures = impl_->consecutive_failures;
  state.cooldown_active = impl_->cooldown_until > std::chrono::steady_clock::now();
  state.degraded_last_call = impl_->degraded_last_call;
  return state;
}

}  // namespace pellet::infer
