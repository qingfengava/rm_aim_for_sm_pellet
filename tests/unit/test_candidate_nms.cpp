#include <gtest/gtest.h>

#include <vector>

#include <opencv2/core/types.hpp>

#include "pellet/imgprocess/candidate_nms.hpp"

namespace {

pellet::imgprocess::Candidate MakeCandidate(
    const cv::Rect& bbox,
    float rank_score,
    int area,
    float brightness) {
  pellet::imgprocess::Candidate c;
  c.bbox = bbox;
  c.rank_score = rank_score;
  c.area = area;
  c.brightness = brightness;
  return c;
}

}  // namespace

TEST(CandidateNmsTest, SuppressesOverlappedLowerRankCandidates) {
  std::vector<pellet::imgprocess::Candidate> candidates;
  candidates.push_back(MakeCandidate(cv::Rect(10, 10, 10, 10), 0.95F, 100, 0.4F));
  candidates.push_back(MakeCandidate(cv::Rect(12, 12, 10, 10), 0.85F, 100, 0.5F));
  candidates.push_back(MakeCandidate(cv::Rect(40, 40, 8, 8), 0.70F, 64, 0.6F));

  const auto kept = pellet::imgprocess::ApplyNms(candidates, 0.30F);
  ASSERT_EQ(kept.size(), 2U);

  EXPECT_NEAR(kept[0].rank_score, 0.95F, 1e-6F);
  EXPECT_EQ(kept[0].bbox.x, 10);
  EXPECT_EQ(kept[0].bbox.y, 10);

  EXPECT_NEAR(kept[1].rank_score, 0.70F, 1e-6F);
  EXPECT_EQ(kept[1].bbox.x, 40);
  EXPECT_EQ(kept[1].bbox.y, 40);
}

TEST(CandidateNmsTest, SupportsPostNmsTopKTruncationOrder) {
  std::vector<pellet::imgprocess::Candidate> candidates;
  candidates.push_back(MakeCandidate(cv::Rect(0, 0, 6, 6), 0.91F, 36, 0.2F));
  candidates.push_back(MakeCandidate(cv::Rect(10, 0, 6, 6), 0.82F, 36, 0.2F));
  candidates.push_back(MakeCandidate(cv::Rect(20, 0, 6, 6), 0.73F, 36, 0.2F));
  candidates.push_back(MakeCandidate(cv::Rect(30, 0, 6, 6), 0.64F, 36, 0.2F));

  auto kept = pellet::imgprocess::ApplyNms(candidates, 0.30F);
  ASSERT_EQ(kept.size(), 4U);

  const std::size_t topk = 2;
  if (kept.size() > topk) {
    kept.resize(topk);
  }

  ASSERT_EQ(kept.size(), 2U);
  EXPECT_NEAR(kept[0].rank_score, 0.91F, 1e-6F);
  EXPECT_NEAR(kept[1].rank_score, 0.82F, 1e-6F);
}
