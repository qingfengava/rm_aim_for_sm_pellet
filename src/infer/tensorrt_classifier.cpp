#include "pellet/infer/tensorrt_classifier.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <wust_vl/common/utils/logger.hpp>

#include "pellet/utils/infer_utils.hpp"
#include "pellet/utils/debug_utils.hpp"

#if defined(PELLET_WITH_TENSORRT)
#include <cuda_runtime.h>
#include <wust_vl/ml_net/tensorrt/tensorrt_net.hpp>
#endif

namespace pellet::infer {
namespace {

constexpr const char* kBackendName = "tensorrt";
constexpr const char* kWarnKeyScratchNotReady = "tensorrt_scratch_not_ready";
constexpr const char* kWarnKeyInferFail = "tensorrt_infer_fail";
constexpr const char* kWarnKeyCooldown = "tensorrt_cooldown";
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

#if defined(PELLET_WITH_TENSORRT)
std::size_t DimsVolume(const nvinfer1::Dims& dims) {
  std::size_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] == 0) {
      return 0;
    }
    if (dims.d[i] > 0) {
      volume *= static_cast<std::size_t>(dims.d[i]);
    }
  }
  return volume;
}
#endif

}  // namespace

struct TensorRtClassifier::Impl {
  bool initialized{false};
  bool runtime_available{false};
  bool debug_log_infer{false};
  int consecutive_failures{0};
  std::chrono::steady_clock::time_point cooldown_until{};
  bool degraded_last_call{false};
#if defined(PELLET_WITH_TENSORRT)
  std::unique_ptr<wust_vl::ml_net::TensorRTNet> net;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream{nullptr};
  int input_width{32};
  int input_height{32};
  int runtime_batch_size{1};
  std::size_t sample_input_count{static_cast<std::size_t>(32 * 32)};
  std::size_t output_count{0};
  std::vector<float> batch_input_scratch;
  std::vector<uint8_t> valid_mask_scratch;
#endif
};

TensorRtClassifier::TensorRtClassifier() : impl_(std::make_unique<Impl>()) {}

TensorRtClassifier::~TensorRtClassifier() = default;

bool TensorRtClassifier::Init(
    const InferenceConfig& config,
    const InferRuntimeOptions& runtime_options) {
  config_ = config;
  impl_->initialized = true;
  impl_->runtime_available = false;
  impl_->debug_log_infer = runtime_options.debug_log_init;
  impl_->consecutive_failures = 0;
  impl_->cooldown_until = std::chrono::steady_clock::time_point{};
  impl_->degraded_last_call = false;

#if defined(PELLET_WITH_TENSORRT)
  impl_->batch_input_scratch.clear();
  impl_->valid_mask_scratch.clear();
  auto GetNormalizedExtension = [](const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return extension;
  };

  const std::string precision = InferToLower(config_.precision);
  if (!InferIsSupportedPrecision(precision)) {
    WUST_ERROR("tensorrt_classifier")
        << "unsupported precision for TensorRT backend: " << config_.precision;
    return false;
  }
  const bool int8_precision = (precision == "int8");
  std::filesystem::path init_path;
  std::filesystem::path engine_path;
  if (!config_.engine_path.empty()) {
    engine_path = std::filesystem::path(config_.engine_path);
    if (GetNormalizedExtension(engine_path) != ".engine") {
      WUST_ERROR("tensorrt_classifier")
          << "engine_path must use .engine suffix, engine_path=" << config_.engine_path;
      return false;
    }
    if (!std::filesystem::exists(engine_path)) {
      WUST_ERROR("tensorrt_classifier")
          << "engine_path not found, engine_path=" << config_.engine_path;
      return false;
    }
    init_path = engine_path;
  } else {
    if (int8_precision) {
      WUST_ERROR("tensorrt_classifier")
          << "unsupported config: precision=int8 requires prebuilt engine_path";
      return false;
    }
    const std::filesystem::path onnx_path(config_.model_path);
    if (onnx_path.empty() || !std::filesystem::exists(onnx_path)) {
      WUST_ERROR("tensorrt_classifier")
          << "onnx model path not found, model_path=" << config_.model_path;
      return false;
    }
    if (GetNormalizedExtension(onnx_path) != ".onnx") {
      WUST_ERROR("tensorrt_classifier")
          << "model_path must be .onnx when engine_path is empty, model_path="
          << config_.model_path;
      return false;
    }
    engine_path = onnx_path;
    engine_path.replace_extension(".engine");
    if (config_.trt_require_prebuilt_engine && !std::filesystem::exists(engine_path)) {
      WUST_ERROR("tensorrt_classifier")
          << "trt_require_prebuilt_engine=1 but engine not found, expected="
          << engine_path.string();
      return false;
    }
    init_path = onnx_path;
  }

  try {
    impl_->input_width = std::max(1, config_.input_width);
    impl_->input_height = std::max(1, config_.input_height);
    impl_->sample_input_count =
        static_cast<std::size_t>(impl_->input_width * impl_->input_height);

    const int requested_batch_size = InferResolveBatchSize(config_.batch_size);
    auto try_init_with_batch = [&](const int batch_size) {
      impl_->net.reset();
      impl_->context.reset();
      impl_->stream = nullptr;
      impl_->output_count = 0U;

      wust_vl::ml_net::TensorRTNet::Params params;
      params.model_path = init_path.string();
      params.input_dims = nvinfer1::Dims4(batch_size, 1, impl_->input_height, impl_->input_width);

      impl_->net = std::make_unique<wust_vl::ml_net::TensorRTNet>();
      if (!impl_->net->init(params)) {
        impl_->net.reset();
        return false;
      }

      auto [input_dims, output_dims] = impl_->net->getInputOutputDims();
      impl_->output_count = DimsVolume(output_dims);
      const int detected_batch = (input_dims.nbDims > 0 && input_dims.d[0] > 0)
                                     ? input_dims.d[0]
                                     : batch_size;
      if (detected_batch != batch_size) {
        impl_->net.reset();
        return false;
      }
      impl_->runtime_batch_size = batch_size;

      impl_->context.reset(impl_->net->getAContext());
      impl_->stream = impl_->net->getStream();
      return impl_->context != nullptr;
    };

    impl_->runtime_available = try_init_with_batch(requested_batch_size);
    if (!impl_->runtime_available && requested_batch_size > 1) {
      impl_->runtime_available = try_init_with_batch(1);
    }
    if (impl_->runtime_available) {
      const std::size_t scratch_count = static_cast<std::size_t>(impl_->runtime_batch_size) *
          impl_->sample_input_count;
      impl_->batch_input_scratch.assign(scratch_count, 0.0F);
      impl_->valid_mask_scratch.assign(
          static_cast<std::size_t>(impl_->runtime_batch_size), static_cast<uint8_t>(0));
      LogInitDebug(
          config_,
          runtime_options,
          init_path.string(),
          impl_->runtime_batch_size > 1 ? "true-batch" : "micro-batch");
    }
  } catch (const std::exception&) {
    WUST_ERROR("tensorrt_classifier") << "TensorRT backend init failed";
    impl_->context.reset();
    impl_->net.reset();
    impl_->runtime_available = false;
  }
#endif

  return impl_->runtime_available;
}

std::vector<float> TensorRtClassifier::Infer(const std::vector<cv::Mat>& rois) {
  impl_->degraded_last_call = false;
  if (!impl_->initialized || !impl_->runtime_available) {
    impl_->degraded_last_call = true;
    return std::vector<float>(rois.size(), 0.0F);
  }

  std::vector<float> scores(rois.size(), 0.0F);

#if defined(PELLET_WITH_TENSORRT)
  if (impl_->net == nullptr || impl_->context == nullptr) {
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

  const std::size_t chunk_size = static_cast<std::size_t>(InferResolveBatchSize(impl_->runtime_batch_size));
  const std::size_t input_elements = chunk_size * impl_->sample_input_count;
  if (impl_->batch_input_scratch.size() < input_elements ||
      impl_->valid_mask_scratch.size() < chunk_size) {
    impl_->degraded_last_call = true;
    if (utils::ShouldLogRateLimited("infer", kWarnKeyScratchNotReady)) {
      WUST_WARN("infer")
          << "backend=" << kBackendName
          << ", scratch buffer not ready, input_scratch=" << impl_->batch_input_scratch.size()
          << ", valid_mask=" << impl_->valid_mask_scratch.size()
          << ", expected_input=" << input_elements
          << ", expected_mask=" << chunk_size;
    }
    return scores;
  }

  try {
    for (std::size_t chunk_begin = 0; chunk_begin < rois.size(); chunk_begin += chunk_size) {
      const std::size_t chunk_count = std::min(chunk_size, rois.size() - chunk_begin);
      std::fill_n(impl_->batch_input_scratch.data(), input_elements, 0.0F);

      std::fill_n(impl_->valid_mask_scratch.data(), chunk_count, static_cast<uint8_t>(0));
      std::size_t valid_count = 0U;
      for (std::size_t i = 0; i < chunk_count; ++i) {
        const bool filled = FillInputTensor(
            rois[chunk_begin + i],
            impl_->batch_input_scratch.data() +
                static_cast<std::ptrdiff_t>(i * impl_->sample_input_count),
            impl_->sample_input_count);
        if (!filled) {
          continue;
        }
        impl_->valid_mask_scratch[i] = 1U;
        ++valid_count;
      }

      if (valid_count == 0U) {
        continue;
      }

      impl_->net->input2Device(impl_->batch_input_scratch.data());
      impl_->net->infer(impl_->net->getInputTensorPtr(), impl_->context.get());
      float* output = impl_->net->output2Host();
      if (impl_->stream != nullptr) {
        cudaStreamSynchronize(impl_->stream);
      }
      if (output == nullptr || impl_->output_count == 0U) {
        continue;
      }
      for (std::size_t i = 0; i < chunk_count; ++i) {
        if (impl_->valid_mask_scratch[i] == 0U) {
          continue;
        }
        scores[chunk_begin + i] =
            ExtractBatchScoreAt(output, impl_->output_count, chunk_size, i);
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
    if (utils::ShouldLogRateLimited("infer", kWarnKeyInferFail)) {
      if (impl_->debug_log_infer) {
        WUST_WARN("infer")
            << "backend=" << kBackendName
            << ", infer failed once, fallback remaining samples to 0.0"
            << ", consecutive_failures=" << impl_->consecutive_failures
            << ", cooldown_active="
            << (impl_->cooldown_until > std::chrono::steady_clock::now() ? 1 : 0)
            << ", err=" << e.what();
      } else {
        WUST_WARN("infer")
            << "backend=" << kBackendName
            << ", infer failed once, fallback remaining samples to 0.0"
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
    if (utils::ShouldLogRateLimited("infer", kWarnKeyInferFail)) {
      WUST_WARN("infer")
          << "backend=" << kBackendName
          << ", infer failed once, fallback remaining samples to 0.0"
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

InferRuntimeState TensorRtClassifier::GetRuntimeState() const {
  InferRuntimeState state;
  state.consecutive_failures = impl_->consecutive_failures;
  state.cooldown_active = impl_->cooldown_until > std::chrono::steady_clock::now();
  state.degraded_last_call = impl_->degraded_last_call;
  return state;
}

}  // namespace pellet::infer
