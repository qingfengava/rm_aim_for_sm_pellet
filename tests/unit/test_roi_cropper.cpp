#include <gtest/gtest.h>

#include <vector>

#include <opencv2/core.hpp>

#include "pellet/imgprocess/roi_cropper.hpp"

TEST(RoiCropperTest, ProducesFixedOutputSize) {
  const cv::Mat gray = cv::Mat::ones(64, 64, CV_8UC1) * 128;

  pellet::imgprocess::Candidate candidate;
  candidate.center = cv::Point2f(32.0F, 32.0F);
  candidate.area = 25;

  pellet::imgprocess::RoiCropConfig cfg;
  cfg.output_size = 32;

  const auto batch = pellet::imgprocess::CropRoiBatch(gray, {candidate}, cfg);

  ASSERT_EQ(batch.patches.size(), 1U);
  EXPECT_EQ(batch.patches[0].rows, 32);
  EXPECT_EQ(batch.patches[0].cols, 32);
  EXPECT_EQ(batch.patches[0].type(), CV_8UC1);
}

TEST(RoiCropperTest, SupportsEdgeCandidateWithPadding) {
  const cv::Mat gray = cv::Mat::ones(64, 64, CV_8UC1) * 128;

  pellet::imgprocess::Candidate candidate;
  candidate.center = cv::Point2f(2.0F, 2.0F);
  candidate.area = 49;
  candidate.motion_score = 0.4F;
  candidate.local_contrast = 0.2F;

  pellet::imgprocess::RoiCropConfig cfg;
  cfg.output_size = 32;
  cfg.min_crop = 16;
  cfg.max_crop = 48;

  const auto batch = pellet::imgprocess::CropRoiBatch(gray, {candidate}, cfg);
  ASSERT_EQ(batch.patches.size(), 1U);
  EXPECT_EQ(batch.patches[0].rows, 32);
  EXPECT_EQ(batch.patches[0].cols, 32);
  EXPECT_EQ(batch.patches[0].type(), CV_8UC1);
}

TEST(RoiCropperTest, FiltersLowQualityCandidate) {
  const cv::Mat gray = cv::Mat::ones(64, 64, CV_8UC1) * 128;

  pellet::imgprocess::Candidate candidate;
  candidate.center = cv::Point2f(32.0F, 32.0F);
  candidate.area = 25;
  candidate.motion_score = 0.01F;
  candidate.local_contrast = 0.0F;

  pellet::imgprocess::RoiCropConfig cfg;
  pellet::imgprocess::RoiCropStats stats;
  const auto batch = pellet::imgprocess::CropRoiBatch(gray, {candidate}, cfg, &stats);

  EXPECT_TRUE(batch.patches.empty());
  EXPECT_EQ(stats.total_candidates, 1);
  EXPECT_EQ(stats.filtered_low_quality, 1);
}
