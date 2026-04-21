#pragma once

#include <vector>

#include "pellet/imgprocess/candidate_extractor.h"

namespace pellet::imgprocess {

struct CandidateFilterConfig {
  int area_min{3};
  int area_max{120};
  float aspect_ratio_max{4.0F};
  float min_circularity{0.0F};
  int max_candidates{16};
};

std::vector<Candidate> FilterAndRankCandidates(
    const std::vector<Candidate>& candidates,
    const CandidateFilterConfig& config);

}  // namespace pellet::imgprocess
