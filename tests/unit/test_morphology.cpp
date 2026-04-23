#include <gtest/gtest.h>

#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "pellet/imgprocess/morphology.hpp"

namespace {

void ExpectMatEqual(const cv::Mat& a, const cv::Mat& b) {
  ASSERT_FALSE(a.empty());
  ASSERT_FALSE(b.empty());
  ASSERT_EQ(a.type(), b.type());
  ASSERT_EQ(a.size(), b.size());
  cv::Mat diff;
  cv::absdiff(a, b, diff);
  EXPECT_EQ(cv::countNonZero(diff), 0);
}

int CountComponents(const cv::Mat& binary_mask) {
  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  const int labels_count =
      cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);
  return std::max(0, labels_count - 1);
}

}  // namespace

TEST(MorphologyTest, OpenRemovesIsolatedNoiseWhenMainTargetExists) {
  cv::Mat mask = cv::Mat::zeros(9, 9, CV_8UC1);
  mask(cv::Rect(2, 2, 5, 5)).setTo(255);
  mask.at<std::uint8_t>(0, 0) = 255;  // isolated noise

  const cv::Mat opened = pellet::imgprocess::ApplyOpen(mask, 3, 1, false);

  EXPECT_EQ(opened.at<std::uint8_t>(0, 0), 0);  // isolated noise removed
  EXPECT_LT(cv::countNonZero(opened), cv::countNonZero(mask));
  EXPECT_GE(cv::countNonZero(opened), 20);       // main target mostly preserved
}

TEST(MorphologyTest, OpenFallsBackWhenForegroundDropsTooMuch) {
  cv::Mat mask = cv::Mat::zeros(7, 7, CV_8UC1);
  mask.at<std::uint8_t>(3, 3) = 255;

  const cv::Mat opened = pellet::imgprocess::ApplyOpen(mask, 3, 1, false);

  // Opening removes this pixel, but fallback should keep original binary mask.
  ExpectMatEqual(opened, mask);
  EXPECT_EQ(cv::countNonZero(opened), 1);
}

TEST(MorphologyTest, CloseFallsBackWhenOverMergeDetected) {
  cv::Mat mask = cv::Mat::ones(5, 5, CV_8UC1) * 255;
  mask.at<std::uint8_t>(2, 2) = 0;  // a tiny hole

  const cv::Mat closed = pellet::imgprocess::ApplyClose(mask, 3, 1, false);

  // Current safety policy should fallback for this case.
  ExpectMatEqual(closed, mask);
}

TEST(MorphologyTest, CloseKeepsStableTwoComponentMask) {
  cv::Mat mask = cv::Mat::zeros(12, 12, CV_8UC1);
  mask(cv::Rect(1, 1, 3, 3)).setTo(255);
  mask(cv::Rect(8, 8, 3, 3)).setTo(255);

  const cv::Mat closed = pellet::imgprocess::ApplyClose(mask, 3, 1, false);
  EXPECT_EQ(CountComponents(closed), 2);
  EXPECT_GT(cv::countNonZero(closed), 0);
}

TEST(MorphologyTest, CloseWithZeroIterationsReturnsBinaryMask) {
  cv::Mat raw = cv::Mat::zeros(3, 3, CV_32FC1);
  raw.at<float>(0, 1) = 2.0F;
  raw.at<float>(1, 1) = 1.0F;
  raw.at<float>(2, 1) = -2.0F;

  const cv::Mat closed = pellet::imgprocess::ApplyClose(raw, 3, 0, false);
  ASSERT_EQ(closed.type(), CV_8UC1);
  EXPECT_EQ(closed.at<std::uint8_t>(0, 1), 255);
  EXPECT_EQ(closed.at<std::uint8_t>(1, 1), 255);
  EXPECT_EQ(closed.at<std::uint8_t>(2, 1), 0);
  EXPECT_EQ(cv::countNonZero(closed), 2);
}
