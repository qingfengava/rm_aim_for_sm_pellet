#pragma once

#include <cstddef>
#include <string>

#include <opencv2/core/mat.hpp>

namespace pellet::infer {

std::string InferToLower(std::string text);

int InferResolveBatchSize(int configured_batch_size);

bool InferIsSupportedPrecision(const std::string& precision);

// ROI uint8→float 归一化，写入 dst。expected_elements 不足时零填充尾部。
bool FillInputTensor(const cv::Mat& roi, float* dst, std::size_t expected_elements);

float ExtractScore(const float* output, std::size_t output_count);

float ExtractBatchScoreAt(
    const float* output,
    std::size_t output_count,
    std::size_t batch_size,
    std::size_t batch_index);

}  // namespace pellet::infer
