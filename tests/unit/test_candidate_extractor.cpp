#include <gtest/gtest.h>

#include <cstdint>

#include <opencv2/core.hpp>

#include "pellet/imgprocess/candidate_extractor.h"

TEST(CandidateExtractorTest, ComputesExtentAndLocalContrastFromMaskPixels) {
  cv::Mat binary_mask = cv::Mat::zeros(32, 32, CV_8UC1);
  binary_mask(cv::Rect(12, 12, 4, 4)).setTo(255);

  cv::Mat gray = cv::Mat::ones(32, 32, CV_8UC1) * 50;
  gray(cv::Rect(12, 12, 4, 4)).setTo(200);

  cv::Mat motion = cv::Mat::zeros(32, 32, CV_8UC1);
  motion(cv::Rect(12, 12, 4, 4)).setTo(255);

  const auto candidates = pellet::imgprocess::ExtractCandidates(binary_mask, gray, motion);
  ASSERT_EQ(candidates.size(), 1U);

  const auto& c = candidates.front();
  EXPECT_EQ(c.area, 16);
  EXPECT_EQ(c.bbox.x, 12);
  EXPECT_EQ(c.bbox.y, 12);
  EXPECT_EQ(c.bbox.width, 4);
  EXPECT_EQ(c.bbox.height, 4);
  EXPECT_NEAR(c.center.x, 13.5F, 1e-3F);
  EXPECT_NEAR(c.center.y, 13.5F, 1e-3F);
  EXPECT_NEAR(c.extent, 1.0F, 1e-6F);
  EXPECT_NEAR(c.local_contrast, (200.0F - 50.0F) / 255.0F, 1e-3F);
  EXPECT_NEAR(c.brightness, 200.0F / 255.0F, 1e-3F);
  EXPECT_NEAR(c.motion_score, 1.0F, 1e-6F);
}

TEST(CandidateExtractorTest, ComputesExtentForPartialFillComponent) {
  cv::Mat binary_mask = cv::Mat::zeros(32, 32, CV_8UC1);
  binary_mask.at<std::uint8_t>(10, 10) = 255;
  binary_mask.at<std::uint8_t>(10, 11) = 255;
  binary_mask.at<std::uint8_t>(11, 10) = 255;

  cv::Mat gray = cv::Mat::ones(32, 32, CV_8UC1) * 80;
  gray.at<std::uint8_t>(10, 10) = 180;
  gray.at<std::uint8_t>(10, 11) = 180;
  gray.at<std::uint8_t>(11, 10) = 180;

  cv::Mat motion = cv::Mat::ones(32, 32, CV_8UC1) * 10;
  motion.at<std::uint8_t>(10, 10) = 120;
  motion.at<std::uint8_t>(10, 11) = 120;
  motion.at<std::uint8_t>(11, 10) = 120;

  const auto candidates = pellet::imgprocess::ExtractCandidates(binary_mask, gray, motion);
  ASSERT_EQ(candidates.size(), 1U);

  const auto& c = candidates.front();
  EXPECT_EQ(c.area, 3);
  EXPECT_EQ(c.bbox.width, 2);
  EXPECT_EQ(c.bbox.height, 2);
  EXPECT_NEAR(c.extent, 0.75F, 1e-6F);
}
