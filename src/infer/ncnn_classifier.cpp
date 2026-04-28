#include "pellet/infer/ncnn_classifier.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <wust_vl/common/utils/logger.hpp>

#if defined(PELLET_WITH_NCNN)
#include <wust_vl/ml_net/ncnn/ncnn_net.hpp>
#endif

namespace pellet::infer {
namespace {

constexpr const char* kBackendName = "ncnn";
constexpr int kInferFailureCooldownThreshold = 3;
constexpr auto kInferFailureCooldown = std::chrono::milliseconds(800);

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

int ResolveBatchSize(const int configured_batch_size) {
  return std::max(1, configured_batch_size);
}

bool IsSupportedPrecision(const std::string& precision) {
  return precision == "fp32" || precision == "int8";
}

void LogInitDebug(
    const InferenceConfig& config,
    const InferRuntimeOptions& runtime_options,
    const std::string& model_resolved_path,
    const char* actual_mode) {
  if (!runtime_options.debug_log_init) {
    return;
  }
  WUST_INFO("infer") << "backend=" << kBackendName
                     << ", precision=" << ToLower(config.precision)
                     << ", batch_size=" << ResolveBatchSize(config.batch_size)
                     << ", model_resolved_path=" << model_resolved_path
                     << ", actual_mode=" << actual_mode;
}

bool EndsWith(const std::string& text, const char* suffix) {
  const std::size_t suffix_len = std::char_traits<char>::length(suffix);
  return text.size() >= suffix_len &&
         text.compare(text.size() - suffix_len, suffix_len, suffix) == 0;
}

std::string DeriveBinPathFromParam(const std::string& param_path) {
  if (EndsWith(param_path, ".param")) {
    return param_path.substr(0, param_path.size() - 6U) + ".bin";
  }
  return std::string();
}

struct NcnnModelPaths {
  std::string param_path;
  std::string bin_path;
};

NcnnModelPaths ResolveNcnnModelPaths(const InferenceConfig& config) {
  const std::string precision = ToLower(config.precision);
  const bool int8_precision = (precision == "int8");

  NcnnModelPaths paths;
  if (int8_precision) {
    // INT8 path must use pre-quantized NCNN artifacts supplied by user.
    paths.param_path = config.ncnn_param_path;
    paths.bin_path = config.ncnn_bin_path;
    if (paths.param_path.empty() && EndsWith(config.int8_model_path, ".param")) {
      paths.param_path = config.int8_model_path;
    }
  } else {
    paths.param_path = config.ncnn_param_path;
    paths.bin_path = config.ncnn_bin_path;
    if (paths.param_path.empty() && EndsWith(config.model_path, ".param")) {
      paths.param_path = config.model_path;
    }
  }

  if (paths.bin_path.empty() && !paths.param_path.empty()) {
    paths.bin_path = DeriveBinPathFromParam(paths.param_path);
  }
  return paths;
}

#if defined(PELLET_WITH_NCNN)
ncnn::Mat CreateInputScratch(const int width, const int height) {
  ncnn::Mat input;
  input.create(width, height);
  return input;
}

bool FillInputTensor(const cv::Mat& roi, ncnn::Mat* input) {
  if (roi.empty() || roi.type() != CV_8UC1 || input == nullptr || input->empty()) {
    return false;
  }

  constexpr float kInv255 = 1.0F / 255.0F;
  if (roi.rows == input->h && roi.cols == input->w) {
    if (roi.isContinuous()) {
      const uint8_t* src = roi.ptr<uint8_t>();
      float* dst = input->row(0);
      const std::size_t total = static_cast<std::size_t>(input->w) * static_cast<std::size_t>(input->h);
      for (std::size_t i = 0; i < total; ++i) {
        dst[i] = static_cast<float>(src[i]) * kInv255;
      }
      return true;
    }
    for (int y = 0; y < input->h; ++y) {
      const uint8_t* src_row = roi.ptr<uint8_t>(y);
      float* dst_row = input->row(y);
      for (int x = 0; x < input->w; ++x) {
        dst_row[x] = static_cast<float>(src_row[x]) * kInv255;
      }
    }
    return true;
  }

  const int copy_h = std::min(input->h, roi.rows);
  const int copy_w = std::min(input->w, roi.cols);
  input->fill(0.0F);
  for (int y = 0; y < copy_h; ++y) {
    const uint8_t* src_row = roi.ptr<uint8_t>(y);
    float* dst_row = input->row(y);
    for (int x = 0; x < copy_w; ++x) {
      dst_row[x] = static_cast<float>(src_row[x]) * kInv255;
    }
  }
  return true;
}

float ExtractScore(const ncnn::Mat& output) {
  if (output.empty() || output.total() == 0U) {
    return 0.0F;
  }
  const float score = output.total() >= 2U ? output[1] : output[0];
  return std::clamp(score, 0.0F, 1.0F);
}
#endif

}  // namespace

struct NcnnClassifier::Impl {
  bool initialized{false};
  bool runtime_available{false};
  bool debug_log_infer{false};
  int consecutive_failures{0};
  std::chrono::steady_clock::time_point cooldown_until{};
#if defined(PELLET_WITH_NCNN)
  std::unique_ptr<wust_vl::ml_net::NCNNNet> net;
  int batch_size{1};
  int input_width{32};
  int input_height{32};
  ncnn::Mat input_scratch;
#endif
};

NcnnClassifier::NcnnClassifier() : impl_(std::make_unique<Impl>()) {}

NcnnClassifier::~NcnnClassifier() = default;

bool NcnnClassifier::Init(
    const InferenceConfig& config,
    const InferRuntimeOptions& runtime_options) {
  config_ = config;
  impl_->initialized = true;
  impl_->runtime_available = false;
  impl_->debug_log_infer = runtime_options.debug_log_init;
  impl_->consecutive_failures = 0;
  impl_->cooldown_until = std::chrono::steady_clock::time_point{};

#if defined(PELLET_WITH_NCNN)
  const std::string precision = ToLower(config_.precision);
  if (!IsSupportedPrecision(precision)) {
    WUST_ERROR("ncnn_classifier")
        << "unsupported precision for NCNN backend: " << config_.precision;
    return false;
  }

  const NcnnModelPaths paths = ResolveNcnnModelPaths(config_);
  const std::string& param_path = paths.param_path;
  const std::string& bin_path = paths.bin_path;
  const bool int8_precision = (precision == "int8");

  if (param_path.empty() || bin_path.empty()) {
    if (int8_precision) {
      WUST_ERROR("ncnn_classifier")
          << "precision=int8 requires pre-quantized ncnn .param/.bin "
          << "(ncnn_param_path/ncnn_bin_path or int8_model_path=.param)";
    }
    return false;
  }
  if (!std::filesystem::exists(param_path) || !std::filesystem::exists(bin_path)) {
    if (int8_precision) {
      WUST_ERROR("ncnn_classifier")
          << "INT8 NCNN artifacts not found, param=" << param_path
          << ", bin=" << bin_path;
    } else {
      WUST_ERROR("ncnn_classifier")
          << "NCNN artifacts not found, param=" << param_path
          << ", bin=" << bin_path;
    }
    return false;
  }

  {
    try {
      impl_->batch_size = ResolveBatchSize(config_.batch_size);
      impl_->input_width = std::max(1, config_.input_width);
      impl_->input_height = std::max(1, config_.input_height);
      impl_->input_scratch = CreateInputScratch(impl_->input_width, impl_->input_height);
      if (impl_->input_scratch.empty()) {
        WUST_ERROR("ncnn_classifier")
            << "failed to allocate input scratch, size="
            << impl_->input_width << "x" << impl_->input_height;
        impl_->runtime_available = false;
        impl_->net.reset();
        return false;
      }

      wust_vl::ml_net::NCNNNet::Params params;
      params.model_path_param = param_path;
      params.model_path_bin = bin_path;
      params.input_name =
          config_.input_blob_name.empty() ? std::string("input") : config_.input_blob_name;
      params.output_name =
          config_.output_blob_name.empty() ? std::string("output") : config_.output_blob_name;
      params.use_vulkan = (ToLower(config_.device) == "gpu" || ToLower(config_.device) == "vulkan");
      params.device_id = 0;
      params.use_light_mode = true;
      params.cpu_threads = std::max(1, config_.num_threads);

      impl_->net = std::make_unique<wust_vl::ml_net::NCNNNet>();
      impl_->net->init(params);
      impl_->runtime_available = true;
      LogInitDebug(
          config_,
          runtime_options,
          param_path + "," + bin_path,
          "micro-batch");
    } catch (const std::exception&) {
      WUST_ERROR("ncnn_classifier") << "NCNN backend init failed";
      impl_->net.reset();
      impl_->runtime_available = false;
    }
  }
#endif

  return impl_->runtime_available;
}

std::vector<float> NcnnClassifier::Infer(const std::vector<cv::Mat>& rois) {
  if (!impl_->initialized || !impl_->runtime_available) {
    return std::vector<float>(rois.size(), 0.0F);
  }

  std::vector<float> scores(rois.size(), 0.0F);

#if defined(PELLET_WITH_NCNN)
  if (impl_->net == nullptr || impl_->input_scratch.empty()) {
    return scores;
  }

  const auto now = std::chrono::steady_clock::now();
  if (impl_->cooldown_until > now) {
    if (impl_->debug_log_infer) {
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

  const std::size_t chunk_size = static_cast<std::size_t>(ResolveBatchSize(impl_->batch_size));
  try {
    for (std::size_t chunk_begin = 0; chunk_begin < rois.size(); chunk_begin += chunk_size) {
      const std::size_t chunk_count = std::min(chunk_size, rois.size() - chunk_begin);
      for (std::size_t i = 0; i < chunk_count; ++i) {
        const std::size_t roi_index = chunk_begin + i;
        const bool filled = FillInputTensor(rois[chunk_begin + i], &impl_->input_scratch);
        if (!filled) {
          continue;
        }
        const ncnn::Mat output = impl_->net->infer(impl_->input_scratch);
        scores[roi_index] = ExtractScore(output);
      }
    }
    impl_->consecutive_failures = 0;
    impl_->cooldown_until = std::chrono::steady_clock::time_point{};
  } catch (const std::exception& e) {
    ++impl_->consecutive_failures;
    if (impl_->consecutive_failures >= kInferFailureCooldownThreshold) {
      impl_->cooldown_until = std::chrono::steady_clock::now() + kInferFailureCooldown;
    }
    if (impl_->debug_log_infer) {
      WUST_WARN("infer")
          << "backend=" << kBackendName
          << ", infer failed once, fallback remaining samples to 0.0"
          << ", consecutive_failures=" << impl_->consecutive_failures
          << ", cooldown_active="
          << (impl_->cooldown_until > std::chrono::steady_clock::now() ? 1 : 0)
          << ", err=" << e.what();
    }
    return scores;
  } catch (...) {
    ++impl_->consecutive_failures;
    if (impl_->consecutive_failures >= kInferFailureCooldownThreshold) {
      impl_->cooldown_until = std::chrono::steady_clock::now() + kInferFailureCooldown;
    }
    if (impl_->debug_log_infer) {
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

}  // namespace pellet::infer
