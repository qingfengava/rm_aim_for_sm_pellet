#include "pellet/imgprocess/candidate_filter.h"

#include <algorithm>

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
    if (candidate.circularity < config.min_circularity) {
      continue;
    }
    filtered.push_back(candidate);
  }

  std::sort(filtered.begin(), filtered.end(), [](const Candidate& a, const Candidate& b) {
    if (a.area != b.area) {
      return a.area > b.area;
    }
    return a.brightness > b.brightness;
  });

  const std::size_t max_candidates = static_cast<std::size_t>(std::max(0, config.max_candidates));
  if (filtered.size() > max_candidates) {
    filtered.resize(max_candidates);
  }

  return filtered;
}

}  // namespace pellet::imgprocess
