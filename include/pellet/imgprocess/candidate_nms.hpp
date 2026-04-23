#pragma once

#include <vector>

#include "pellet/imgprocess/candidate_extractor.hpp"

namespace pellet::imgprocess {

std::vector<Candidate> ApplyNms(
    const std::vector<Candidate>& candidates,
    float iou_thresh);

}  // namespace pellet::imgprocess
