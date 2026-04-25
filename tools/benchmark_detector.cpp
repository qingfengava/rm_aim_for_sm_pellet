#include <chrono>

#include <opencv2/core.hpp>
#include <wust_vl/common/utils/logger.hpp>

#include "pellet/detector/detector.hpp"

int main() {
  wust_vl::initLogger("INFO", "logs", true, false, true);

  pellet::PelletConfig config;
  config.inference.backend = "mock";
  config.inference.positive_threshold = 0.7F;

  pellet::PelletDetector detector(config);
  const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

  const auto begin = std::chrono::steady_clock::now();
  constexpr int kIters = 200;

  std::size_t total = 0;
  for (int i = 0; i < kIters; ++i) {
    const auto detections = detector.ProcessFrame(frame, static_cast<uint32_t>(i), i * 8);
    total += detections.size();
  }

  const auto end = std::chrono::steady_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

  WUST_MAIN("benchmark")
      << "benchmark_detector: " << kIters << " frames in " << ms << " ms, detections=" << total;
  return 0;
}
