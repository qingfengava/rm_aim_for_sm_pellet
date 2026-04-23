#include "pellet/imgprocess/candidate_filter.hpp"

#include <algorithm>
#include <cmath>

namespace pellet::imgprocess {

std::vector<Candidate> FilterAndRankCandidates(
    const std::vector<Candidate>& candidates,
    const CandidateFilterConfig& config) {
  std::vector<Candidate> filtered;
  filtered.reserve(candidates.size());

  for (const auto& candidate : candidates) {
    if (candidate.area < config.area_min || candidate.area > config.area_max) {
      continue;
    }
    if (candidate.aspect_ratio > config.aspect_ratio_max) {
      continue;
    }
    if (candidate.extent < config.extent_min) {
      continue;
    }
    if (candidate.local_contrast < config.contrast_min) {
      continue;
    }
    if (candidate.motion_score < config.motion_score_min) {
      continue;
    }
    if (candidate.circularity < config.min_circularity) {
      continue;
    }
    filtered.push_back(candidate);
  }

  std::sort(filtered.begin(), filtered.end(), [](const Candidate& a, const Candidate& b) {
    constexpr float kScoreEps = 1e-6F;
    if (std::fabs(a.rank_score - b.rank_score) > kScoreEps) {
      return a.rank_score > b.rank_score;
    }
    if (a.area != b.area) {
      return a.area > b.area;
    }
    return a.brightness > b.brightness;
  });

  //限制最终候选数量
  const std::size_t max_candidates = static_cast<std::size_t>(std::max(0, config.max_candidates));
  if (filtered.size() > max_candidates) {
    filtered.resize(max_candidates);
  }

  return filtered;
}

}  // namespace pellet::imgprocess
