#include "pellet/utils/infer_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>

namespace pellet::infer {

std::string InferToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

int InferResolveBatchSize(int configured_batch_size) {
  return std::max(1, configured_batch_size);
}

bool InferIsSupportedPrecision(const std::string& precision) {
  return precision == "fp32" || precision == "int8";
}

bool FillInputTensor(const cv::Mat& roi, float* dst, std::size_t expected_elements) {
  if (roi.empty() || roi.type() != CV_8UC1 || dst == nullptr || expected_elements == 0U) {
    return false;
  }

  const std::size_t base_elements = static_cast<std::size_t>(roi.total());
  if (base_elements == 0U) {
    return false;
  }

  constexpr float kInv255 = 1.0F / 255.0F;

  if (expected_elements == base_elements) {
    if (roi.isContinuous()) {
      const std::uint8_t* src = roi.ptr<std::uint8_t>();
      for (std::size_t i = 0; i < base_elements; ++i) {
        dst[i] = static_cast<float>(src[i]) * kInv255;
      }
      return true;
    }
    std::size_t idx = 0U;
    for (int y = 0; y < roi.rows; ++y) {
      const std::uint8_t* row = roi.ptr<std::uint8_t>(y);
      for (int x = 0; x < roi.cols; ++x) {
        dst[idx++] = static_cast<float>(row[x]) * kInv255;
      }
    }
    return true;
  }

  // 尺寸不匹配：先零填充再拷贝有效数据，不隐式平铺
  std::fill_n(dst, expected_elements, 0.0F);
  const std::size_t copy_count = std::min(expected_elements, base_elements);
  if (roi.isContinuous()) {
    const std::uint8_t* src = roi.ptr<std::uint8_t>();
    for (std::size_t i = 0; i < copy_count; ++i) {
      dst[i] = static_cast<float>(src[i]) * kInv255;
    }
    return true;
  }
  std::size_t idx = 0U;
  for (int y = 0; y < roi.rows && idx < copy_count; ++y) {
    const std::uint8_t* row = roi.ptr<std::uint8_t>(y);
    for (int x = 0; x < roi.cols && idx < copy_count; ++x) {
      dst[idx++] = static_cast<float>(row[x]) * kInv255;
    }
  }
  return true;
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

}  // namespace pellet::infer
