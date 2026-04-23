#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "pellet/detector/detector.hpp"

TEST(DetectorE2ETest, ProcessFrameRuns) {
  pellet::PelletConfig config;
  config.inference.backend = "mock";
  config.inference.positive_threshold = 0.1F;

  pellet::PelletDetector detector(config);
  const cv::Mat frame = cv::Mat::zeros(64, 64, CV_8UC3);

  const auto detections = detector.ProcessFrame(frame, 1, 1000);
  EXPECT_GE(detections.size(), 0U);
}
