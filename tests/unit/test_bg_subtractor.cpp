#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "pellet/imgprocess/bg_subtractor.hpp"

using namespace pellet::imgprocess;

TEST(BgSubtractor, FirstFrameReturnsZeroMotionResponse) {
  BgSubtractConfig cfg;
  cfg.downsample = 1;
  BgSubtractor det(cfg);

  cv::Mat frame(240, 320, CV_8UC1, cv::Scalar(128));
  BgSubtractResult result = det.Apply(frame);

  EXPECT_FALSE(result.binary_mask.empty());
  EXPECT_EQ(result.binary_mask.size(), frame.size());
  // 第一帧 motion_response 应为全零
  EXPECT_EQ(cv::countNonZero(result.motion_response), 0);
}

TEST(BgSubtractor, MovingObjectDetected) {
  BgSubtractConfig cfg;
  cfg.backend = "knn";
  cfg.downsample = 1;
  cfg.history = 20;
  BgSubtractor det(cfg);

  cv::Mat bg(240, 320, CV_8UC1, cv::Scalar(128));

  for (int i = 0; i < 30; ++i) {
    det.Apply(bg);
  }

  cv::Mat fg = bg.clone();
  cv::circle(fg, cv::Point(160, 120), 6, cv::Scalar(200), -1);
  BgSubtractResult result = det.Apply(fg);

  const int fg_count = cv::countNonZero(result.binary_mask);
  EXPECT_GT(fg_count, 0);
}

TEST(BgSubtractor, DownsampleWorks) {
  BgSubtractConfig cfg;
  cfg.downsample = 2;
  BgSubtractor det(cfg);

  cv::Mat frame(240, 320, CV_8UC1, cv::Scalar(128));
  BgSubtractResult result = det.Apply(frame);

  EXPECT_EQ(result.binary_mask.size(), frame.size());
}

TEST(BgSubtractor, MotionResponseComputed) {
  BgSubtractConfig cfg;
  cfg.downsample = 1;
  BgSubtractor det(cfg);

  cv::Mat frame1(240, 320, CV_8UC1, cv::Scalar(128));
  det.Apply(frame1);

  cv::Mat frame2 = frame1.clone();
  cv::rectangle(frame2, cv::Rect(100, 80, 20, 20), cv::Scalar(200), -1);
  BgSubtractResult result = det.Apply(frame2);

  EXPECT_GT(cv::countNonZero(result.motion_response), 0);
}

TEST(BgSubtractor, ResetClearsModel) {
  BgSubtractConfig cfg;
  cfg.downsample = 1;
  BgSubtractor det(cfg);

  cv::Mat frame(240, 320, CV_8UC1, cv::Scalar(128));
  for (int i = 0; i < 10; ++i) {
    det.Apply(frame);
  }

  det.Reset();
  BgSubtractResult result = det.Apply(frame);
  EXPECT_EQ(cv::countNonZero(result.motion_response), 0);
}
