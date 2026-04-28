#include "pellet/infer/tensorrt_classifier.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <wust_vl/common/utils/logger.hpp>

#if defined(PELLET_WITH_TENSORRT)
#include <cuda_runtime.h>
#include <wust_vl/ml_net/tensorrt/tensorrt_net.hpp>
#endif

namespace pellet::infer {
namespace {

constexpr const char* kBackendName = "tensorrt";

int ResolveBatchSize(const int configured_batch_size) {
  return std::max(1, configured_batch_size);
}

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
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

bool FillInputTensor(const cv::Mat& roi, float* dst, std::size_t expected_elements) {
  if (roi.empty() || roi.type() != CV_8UC1 || dst == nullptr || expected_elements == 0U) {
    return false; //要求 ROI 必须 CV_8UC1
  }

  const std::size_t base_elements = static_cast<std::size_t>(roi.total());
  if (base_elements == 0U) {
    return false;
  }

  constexpr float kInv255 = 1.0F / 255.0F; //归一化到 [0,1]
  auto copy_plane = [&](float* out) {
    if (roi.isContinuous()) {
      const uint8_t* src = roi.ptr<uint8_t>();
      for (std::size_t i = 0; i < base_elements; ++i) {
        out[i] = static_cast<float>(src[i]) * kInv255;
      }
      return;
    }
    std::size_t index = 0U;
    for (int y = 0; y < roi.rows; ++y) {
      const uint8_t* row = roi.ptr<uint8_t>(y);
      for (int x = 0; x < roi.cols; ++x) {
        out[index++] = static_cast<float>(row[x]) * kInv255;
      }
    }
  };

  if (expected_elements == base_elements) {
    copy_plane(dst);
    return true;
  }
  if (expected_elements > base_elements && expected_elements % base_elements == 0U) {
    const std::size_t repeat = expected_elements / base_elements;
    for (std::size_t i = 0; i < repeat; ++i) {
      copy_plane(dst + static_cast<std::ptrdiff_t>(i * base_elements));
    }
    return true;
  }

  std::fill_n(dst, expected_elements, 0.0F);
  const std::size_t copy_count = std::min(expected_elements, base_elements);
  if (roi.isContinuous()) {
    const uint8_t* src = roi.ptr<uint8_t>();
    for (std::size_t i = 0; i < copy_count; ++i) {
      dst[i] = static_cast<float>(src[i]) * kInv255;
    }
    return true;
  }
  std::size_t index = 0U;
  for (int y = 0; y < roi.rows && index < copy_count; ++y) {
    const uint8_t* row = roi.ptr<uint8_t>(y);
    for (int x = 0; x < roi.cols && index < copy_count; ++x) {
      dst[index++] = static_cast<float>(row[x]) * kInv255;
    }
  }
  return true;
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

float ExtractScore(const float* output, std::size_t output_count) {
  if (output == nullptr || output_count == 0) {
    return 0.0F;
  }
  const float score = output_count >= 2 ? output[1] : output[0];
  return std::clamp(score, 0.0F, 1.0F);
}
  //取分
float ExtractBatchScoreAt(
    const float* output,
    std::size_t output_count,
    std::size_t batch_size,
    std::size_t batch_index) {
  batch_size = std::max<std::size_t>(1, batch_size);
  if (output == nullptr || output_count == 0) {
    return 0.0F;
  }
  if (batch_size == 1U) {
    return batch_index == 0U ? ExtractScore(output, output_count) : 0.0F;
  }
  if (output_count < batch_size) {
    return ExtractScore(output, output_count);
  }
  const std::size_t stride = std::max<std::size_t>(1, output_count / batch_size);
  const std::size_t start = batch_index * stride;
  if (start >= output_count) {
    return 0.0F;
  }
  const std::size_t remaining = output_count - start;
  const std::size_t sample_count = std::min(stride, remaining);
  return ExtractScore(output + static_cast<std::ptrdiff_t>(start), sample_count);
}

}  // namespace

struct TensorRtClassifier::Impl {
  bool initialized{false};
  bool runtime_available{false};
  bool debug_log_infer{false};
#if defined(PELLET_WITH_TENSORRT)
  std::unique_ptr<wust_vl::ml_net::TensorRTNet> net;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream{nullptr};
  int input_width{32};
  int input_height{32};
  int runtime_batch_size{1};
  std::size_t sample_input_count{static_cast<std::size_t>(32 * 32)};
  std::size_t output_count{0};
  std::vector<float> batch_input_scratch;//输入缓存
  std::vector<uint8_t> valid_mask_scratch;//有效ROI缓存
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

  const std::string precision = ToLower(config_.precision);
  if (!IsSupportedPrecision(precision)) {
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
      // Current wust_vl auto-build path does not include INT8 calibration setup.
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

    const int requested_batch_size = ResolveBatchSize(config_.batch_size);
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
      impl_->runtime_available = try_init_with_batch(1);//batch fallback to 1
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
  if (!impl_->initialized || !impl_->runtime_available) {
    return std::vector<float>(rois.size(), 0.0F);
  }

  std::vector<float> scores(rois.size(), 0.0F);

#if defined(PELLET_WITH_TENSORRT)
  if (impl_->net == nullptr || impl_->context == nullptr) {
    return scores;
  }

  const std::size_t chunk_size = static_cast<std::size_t>(ResolveBatchSize(impl_->runtime_batch_size));
  const std::size_t input_elements = chunk_size * impl_->sample_input_count;
  if (impl_->batch_input_scratch.size() < input_elements ||
      impl_->valid_mask_scratch.size() < chunk_size) {
    if (impl_->debug_log_infer) {
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
        impl_->valid_mask_scratch[i] = 1U;//标记有效 ROI
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
  } catch (const std::exception& e) {
    if (impl_->debug_log_infer) {
      WUST_WARN("infer")
          << "backend=" << kBackendName
          << ", infer failed once, fallback remaining samples to 0.0, err=" << e.what();
    }
    return scores;
  } catch (...) {
    if (impl_->debug_log_infer) {
      WUST_WARN("infer")
          << "backend=" << kBackendName
          << ", infer failed once, fallback remaining samples to 0.0";
    }
    return scores;
  }
  return scores;
#endif

  return scores;
}

}  // namespace pellet::infer
