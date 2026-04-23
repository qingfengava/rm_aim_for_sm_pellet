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
}
