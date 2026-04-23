#include <gtest/gtest.h>

#include <vector>

#include "pellet/imgprocess/candidate_filter.hpp"

TEST(CandidateFilterTest, AppliesAreaLimitAndRankingByRankScore) {
  std::vector<pellet::imgprocess::Candidate> candidates;

  pellet::imgprocess::Candidate c1;
  c1.area = 20;
  c1.brightness = 0.4F;
  c1.extent = 0.30F;
  c1.local_contrast = 0.20F;
  c1.motion_score = 0.30F;
  c1.rank_score = 0.40F;
  candidates.push_back(c1);

  pellet::imgprocess::Candidate c2;
  c2.area = 50;
  c2.brightness = 0.2F;
  c2.extent = 0.50F;
  c2.local_contrast = 0.30F;
  c2.motion_score = 0.60F;
  c2.rank_score = 0.95F;
  candidates.push_back(c2);

  pellet::imgprocess::Candidate c3;
  c3.area = 30;
  c3.brightness = 0.8F;
  c3.extent = 0.10F;  // filtered by extent
  c3.local_contrast = 0.40F;
  c3.motion_score = 0.70F;
  c3.rank_score = 0.99F;
  candidates.push_back(c3);

  pellet::imgprocess::CandidateFilterConfig cfg;
  cfg.area_min = 10;
  cfg.area_max = 100;
  cfg.extent_min = 0.20F;
  cfg.contrast_min = 0.10F;
  cfg.motion_score_min = 0.20F;
  cfg.max_candidates = 1;

  const auto filtered = pellet::imgprocess::FilterAndRankCandidates(candidates, cfg);
  ASSERT_EQ(filtered.size(), 1U);
  EXPECT_FLOAT_EQ(filtered.front().rank_score, 0.95F);
  EXPECT_EQ(filtered.front().area, 50);
}

TEST(CandidateFilterTest, FiltersLowContrastReflectiveCandidates) {
  std::vector<pellet::imgprocess::Candidate> candidates;

  pellet::imgprocess::Candidate target;
  target.area = 24;
  target.aspect_ratio = 1.3F;
  target.extent = 0.50F;
  target.local_contrast = 0.22F;
  target.motion_score = 0.45F;
  target.circularity = 0.75F;
  target.rank_score = 0.80F;
  candidates.push_back(target);

  pellet::imgprocess::Candidate reflective;
  reflective.area = 24;
  reflective.aspect_ratio = 1.2F;
  reflective.extent = 0.52F;
  reflective.local_contrast = 0.02F;  // reflective highlight: bright but low local contrast
  reflective.motion_score = 0.70F;
  reflective.circularity = 0.80F;
  reflective.rank_score = 0.92F;
  candidates.push_back(reflective);

  pellet::imgprocess::CandidateFilterConfig cfg;
  cfg.area_min = 5;
  cfg.area_max = 120;
  cfg.aspect_ratio_max = 4.0F;
  cfg.extent_min = 0.20F;
  cfg.contrast_min = 0.10F;
  cfg.motion_score_min = 0.20F;
  cfg.min_circularity = 0.0F;
  cfg.max_candidates = 8;

  const auto filtered = pellet::imgprocess::FilterAndRankCandidates(candidates, cfg);
  ASSERT_EQ(filtered.size(), 1U);
  EXPECT_NEAR(filtered.front().local_contrast, 0.22F, 1e-6F);
}
