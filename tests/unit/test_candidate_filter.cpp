#include <gtest/gtest.h>

#include <vector>

#include "pellet/imgprocess/candidate_filter.h"

TEST(CandidateFilterTest, AppliesAreaLimitAndRanking) {
  std::vector<pellet::imgprocess::Candidate> candidates;

  pellet::imgprocess::Candidate c1;
  c1.area = 20;
  c1.brightness = 0.4F;
  candidates.push_back(c1);

  pellet::imgprocess::Candidate c2;
  c2.area = 50;
  c2.brightness = 0.2F;
  candidates.push_back(c2);

  pellet::imgprocess::CandidateFilterConfig cfg;
  cfg.area_min = 10;
  cfg.area_max = 100;
  cfg.max_candidates = 1;

  const auto filtered = pellet::imgprocess::FilterAndRankCandidates(candidates, cfg);
  ASSERT_EQ(filtered.size(), 1U);
  EXPECT_EQ(filtered.front().area, 50);
}
