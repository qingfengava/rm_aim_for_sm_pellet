#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "pellet/imgprocess/three_frame_diff.h"

TEST(ThreeFrameDiffTest, WarmupReturnsZeroMask) {
  pellet::imgprocess::ThreeFrameDiff diff;

  const cv::Mat frame = cv::Mat::zeros(32, 32, CV_8UC1);
  const cv::Mat out1 = diff.Apply(frame, 30);
  const cv::Mat out2 = diff.Apply(frame, 30);

  EXPECT_EQ(cv::countNonZero(out1), 0);
  EXPECT_EQ(cv::countNonZero(out2), 0);
}
