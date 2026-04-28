#include "pellet/infer/onnx_classifier.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <wust_vl/common/utils/logger.hpp>

#if defined(PELLET_WITH_ORT)
#include <onnxruntime_cxx_api.h>
#endif

namespace pellet::infer {
namespace {

constexpr const char* kBackendName = "onnxruntime";

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

std::string ResolveModelPath(const InferenceConfig& config) {
  const std::string precision = ToLower(config.precision);
  if (precision == "int8" && !config.int8_model_path.empty() &&
      std::filesystem::exists(config.int8_model_path)) {
    return config.int8_model_path;
  }
  return config.model_path;
}

bool FillInputTensor(
    const cv::Mat& roi,
    float* dst,
    std::size_t expected_elements) {
  if (roi.empty() || roi.type() != CV_8UC1 || dst == nullptr || expected_elements == 0U) {
    return false;
  }

  const std::size_t base_elements = static_cast<std::size_t>(roi.total());
  if (base_elements == 0U) {
    return false;
  }

  constexpr float kInv255 = 1.0F / 255.0F; //归一化
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

std::size_t CountElements(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  std::size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim == 0) {
      return 0;
    }
    if (dim > 0) {
      count *= static_cast<std::size_t>(dim);
    }
  }
  return count;
}

float ExtractScore(const float* output, std::size_t output_count) {
  if (output == nullptr || output_count == 0) {
    return 0.0F;
  }
  const float score = output_count >= 2 ? output[1] : output[0];
  return std::clamp(score, 0.0F, 1.0F);
}

float ExtractBatchScoreAt(
    const float* output,
    std::size_t output_count,
    std::size_t batch_size,
    std::size_t batch_index) {
  batch_size = std::max<std::size_t>(1, batch_size);
  if (output == nullptr || output_count == 0) {
    return 0.0F;
  }

  if (batch_size == 1) {
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

#if defined(PELLET_WITH_ORT)
void ConfigureOrtProvider(const std::string& device, Ort::SessionOptions* session_options) {
  if (session_options == nullptr) {
    return;
  }
  const std::string normalized = ToLower(device);
  if (normalized == "gpu" || normalized == "cuda") {
    session_options->AppendExecutionProvider_CUDA({});
    return;
  }
  if (normalized == "trt" || normalized == "tensorrt") {
    session_options->AppendExecutionProvider_TensorRT({});
    return;
  }
  if (normalized == "openvino" || normalized == "ov") {
    OrtOpenVINOProviderOptions options;
    options.device_type = "CPU_FP32";
    session_options->AppendExecutionProvider_OpenVINO(options);
    return;
  }
}

std::vector<int64_t> ResolveInputShape(
    const std::vector<int64_t>& raw_shape,
    int input_width,
    int input_height) {
  if (raw_shape.size() != 4U) {
    return {1, 1, input_height, input_width};
  }
  std::vector<int64_t> shape = raw_shape;
  shape[0] = (shape[0] > 0) ? shape[0] : 1;  // N
  shape[1] = 1;                               // C
  shape[2] = static_cast<int64_t>(input_height);  // H
  shape[3] = static_cast<int64_t>(input_width);   // W
  return shape;
}

#endif

}  // namespace

struct OnnxClassifier::Impl {
  bool initialized{false};
  bool runtime_available{false};
  bool debug_log_infer{false};
#if defined(PELLET_WITH_ORT)
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
  std::unique_ptr<Ort::MemoryInfo> memory_info;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> run_shape_scratch;
  std::vector<float> batch_input_scratch;
  std::vector<uint8_t> valid_mask_scratch;
  std::string input_name;
  std::string output_name;
  int input_width{32};
  int input_height{32};
  int batch_size{1};
  int64_t model_batch_dim{1};
  bool dynamic_batch_allowed{false};
  std::size_t input_element_count{static_cast<std::size_t>(32 * 32)};
  std::size_t sample_element_count{static_cast<std::size_t>(32 * 32)};
  std::size_t output_count{0};
#endif
};

OnnxClassifier::OnnxClassifier() : impl_(std::make_unique<Impl>()) {}

OnnxClassifier::~OnnxClassifier() = default;

bool OnnxClassifier::Init(
    const InferenceConfig& config,
    const InferRuntimeOptions& runtime_options) {
  config_ = config;
  impl_->initialized = true;
  impl_->runtime_available = false;
  impl_->debug_log_infer = runtime_options.debug_log_init;

#if defined(PELLET_WITH_ORT)
  impl_->memory_info.reset();
  impl_->run_shape_scratch.clear();
  impl_->batch_input_scratch.clear();
  impl_->valid_mask_scratch.clear();
  const std::string precision = ToLower(config_.precision);
  if (!IsSupportedPrecision(precision)) {
    WUST_ERROR("onnx_classifier") << "unsupported precision for ONNX backend: " << config_.precision;
    return false;
  }

  const std::string model_path = ResolveModelPath(config_);
  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    WUST_ERROR("onnx_classifier")
        << "model path not found, model_resolved_path=" << model_path;
    return false;
  }

  try {
    impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "pellet_onnx_classifier");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(std::max(1, config_.num_threads));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ConfigureOrtProvider(config_.device, &session_options);

    impl_->session = std::make_unique<Ort::Session>(*impl_->env, model_path.c_str(), session_options);
    impl_->input_width = std::max(1, config_.input_width);
    impl_->input_height = std::max(1, config_.input_height);
    impl_->batch_size = ResolveBatchSize(config_.batch_size);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_alloc =
        impl_->session->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_alloc =
        impl_->session->GetOutputNameAllocated(0, allocator);
    impl_->input_name = input_name_alloc ? input_name_alloc.get() : "";
    impl_->output_name = output_name_alloc ? output_name_alloc.get() : "";
    if (impl_->input_name.empty() || impl_->output_name.empty()) {
      WUST_ERROR("onnx_classifier") << "failed to resolve input/output tensor names";
      impl_->session.reset();
      impl_->env.reset();
      return false;
    }

    const Ort::TypeInfo input_type_info = impl_->session->GetInputTypeInfo(0);
    const auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> raw_input_shape = input_tensor_info.GetShape();
    impl_->dynamic_batch_allowed = !raw_input_shape.empty() && raw_input_shape[0] <= 0;
    impl_->input_shape = ResolveInputShape(
        raw_input_shape,
        impl_->input_width,
        impl_->input_height);
    impl_->model_batch_dim = (!impl_->input_shape.empty() && impl_->input_shape[0] > 0)
                                 ? impl_->input_shape[0]
                                 : 1;
    impl_->input_element_count = CountElements(impl_->input_shape);
    if (impl_->input_element_count == 0) {
      impl_->input_element_count = static_cast<std::size_t>(
          std::max(1, impl_->input_width * impl_->input_height));
    }
    impl_->sample_element_count = impl_->input_element_count;
    if (impl_->model_batch_dim > 1 &&
        impl_->input_element_count % static_cast<std::size_t>(impl_->model_batch_dim) == 0U) {
      impl_->sample_element_count /=
          static_cast<std::size_t>(impl_->model_batch_dim);
    }
    impl_->run_shape_scratch = impl_->input_shape;
    impl_->memory_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    const Ort::TypeInfo output_type_info = impl_->session->GetOutputTypeInfo(0);
    const auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    impl_->output_count = CountElements(output_tensor_info.GetShape());

    const std::size_t configured_chunk_size =
        static_cast<std::size_t>(ResolveBatchSize(impl_->batch_size));
    const std::size_t static_model_batch =
        static_cast<std::size_t>(std::max<int64_t>(1, impl_->model_batch_dim));
    const bool fixed_batch_model = !impl_->dynamic_batch_allowed;
    const std::size_t chunk_size =
        fixed_batch_model ? static_model_batch
                          : configured_chunk_size;
    const std::size_t max_run_batch =
        fixed_batch_model ? static_model_batch
                          : configured_chunk_size;
    const std::size_t max_input_elements = max_run_batch * impl_->sample_element_count;
    if (chunk_size == 0U || max_input_elements == 0U) {
      WUST_ERROR("onnx_classifier")
          << "invalid ONNX runtime buffer shape, chunk_size=" << chunk_size
          << ", max_input_elements=" << max_input_elements;
      impl_->memory_info.reset();
      impl_->session.reset();
      impl_->env.reset();
      impl_->runtime_available = false;
      return false;
    }
    impl_->batch_input_scratch.assign(max_input_elements, 0.0F);
    impl_->valid_mask_scratch.assign(chunk_size, static_cast<uint8_t>(0));

    impl_->runtime_available = true;
    const bool use_true_batch_mode = impl_->dynamic_batch_allowed || impl_->model_batch_dim > 1;
    if (runtime_options.debug_log_init &&
        !impl_->dynamic_batch_allowed &&
        impl_->model_batch_dim > 0 &&
        static_cast<std::size_t>(ResolveBatchSize(config_.batch_size)) <
            static_cast<std::size_t>(impl_->model_batch_dim)) {
      WUST_WARN("onnx_classifier")
          << "configured batch_size=" << ResolveBatchSize(config_.batch_size)
          << " is smaller than fixed model batch=" << impl_->model_batch_dim
          << ", forcing chunk_size/run_batch to model batch";
    }
    LogInitDebug(
        config_,
        runtime_options,
        model_path,
        use_true_batch_mode ? "true-batch" : "micro-batch");
  } catch (const std::exception&) {
    WUST_ERROR("onnx_classifier") << "ONNX backend init failed";
    impl_->memory_info.reset();
    impl_->session.reset();
    impl_->env.reset();
    impl_->runtime_available = false;
  }
#endif

  return impl_->runtime_available;
}

std::vector<float> OnnxClassifier::Infer(const std::vector<cv::Mat>& rois) {
  if (!impl_->initialized || !impl_->runtime_available) {
    return std::vector<float>(rois.size(), 0.0F);
  }

  std::vector<float> scores(rois.size(), 0.0F);

#if defined(PELLET_WITH_ORT)
  if (impl_->session == nullptr || impl_->input_shape.empty() ||
      impl_->input_name.empty() || impl_->output_name.empty() ||
      impl_->memory_info == nullptr) {
    return scores;
  }

  const char* input_names[] = {impl_->input_name.c_str()};
  const char* output_names[] = {impl_->output_name.c_str()};
  const std::size_t configured_chunk_size =
      static_cast<std::size_t>(ResolveBatchSize(impl_->batch_size));
  const std::size_t static_model_batch =
      static_cast<std::size_t>(std::max<int64_t>(1, impl_->model_batch_dim));
  const bool fixed_batch_model = !impl_->dynamic_batch_allowed;
  const std::size_t chunk_size =
      fixed_batch_model ? static_model_batch
                        : configured_chunk_size;
  const std::size_t max_run_batch =
      fixed_batch_model ? static_model_batch
                        : configured_chunk_size;
  const std::size_t max_input_elements = max_run_batch * impl_->sample_element_count;
  if (impl_->batch_input_scratch.size() < max_input_elements ||
      impl_->valid_mask_scratch.size() < chunk_size) {
    if (impl_->debug_log_infer) {
      WUST_WARN("infer")
          << "backend=" << kBackendName
          << ", scratch buffer not ready, input_scratch=" << impl_->batch_input_scratch.size()
          << ", valid_mask=" << impl_->valid_mask_scratch.size()
          << ", expected_input=" << max_input_elements
          << ", expected_mask=" << chunk_size;
    }
    return scores;
  }
  Ort::RunOptions run_options;

  try {
    for (std::size_t chunk_begin = 0; chunk_begin < rois.size(); chunk_begin += chunk_size) {
      const std::size_t chunk_count = std::min(chunk_size, rois.size() - chunk_begin);
      const std::size_t run_batch = fixed_batch_model ? static_model_batch : chunk_count;
      const std::size_t input_elements = run_batch * impl_->sample_element_count;
      std::fill_n(impl_->batch_input_scratch.data(), input_elements, 0.0F);
      std::fill_n(impl_->valid_mask_scratch.data(), chunk_count, static_cast<uint8_t>(0));
      std::size_t valid_count = 0U;
      for (std::size_t i = 0; i < chunk_count; ++i) {
        const bool filled = FillInputTensor(
            rois[chunk_begin + i],
            impl_->batch_input_scratch.data() +
                static_cast<std::ptrdiff_t>(i * impl_->sample_element_count),
            impl_->sample_element_count);
        if (!filled) {
          continue;
        }
        impl_->valid_mask_scratch[i] = 1U;
        ++valid_count;
      }
      if (valid_count == 0U) {
        continue;
      }

      if (impl_->run_shape_scratch.size() != impl_->input_shape.size()) {
        impl_->run_shape_scratch = impl_->input_shape;
      }
      if (!impl_->run_shape_scratch.empty()) {
        impl_->run_shape_scratch[0] = static_cast<int64_t>(run_batch);
      }
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          *impl_->memory_info,
          impl_->batch_input_scratch.data(),
          input_elements,
          impl_->run_shape_scratch.data(),
          impl_->run_shape_scratch.size());
      std::vector<Ort::Value> output_tensors = impl_->session->Run(
          run_options,
          input_names,
          &input_tensor,
          1,
          output_names,
          1);
      if (output_tensors.empty()) {
        continue;
      }

      const Ort::Value& output = output_tensors.front();
      if (!output.IsTensor()) {
        continue;
      }
      const Ort::TensorTypeAndShapeInfo tensor_info = output.GetTensorTypeAndShapeInfo();
      if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        continue;
      }
      const std::size_t output_count = tensor_info.GetElementCount() == 0U
                                           ? impl_->output_count
                                           : tensor_info.GetElementCount();
      const float* output_data = output.GetTensorData<float>();
      for (std::size_t i = 0; i < chunk_count; ++i) {
        if (impl_->valid_mask_scratch[i] == 0U) {
          continue;
        }
        scores[chunk_begin + i] =
            ExtractBatchScoreAt(output_data, output_count, run_batch, i);
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
