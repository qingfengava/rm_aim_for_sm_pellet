#pragma once

#include <vector>

#include "pellet/imgprocess/candidate_extractor.hpp"

namespace pellet::imgprocess {

std::vector<Candidate> ApplyNms(
    const std::vector<Candidate>& candidates,
    float iou_thresh);

// candidates 已按 rank_score 降序排列时可跳过内部排序
std::vector<Candidate> ApplyNmsPreSorted(
    const std::vector<Candidate>& candidates,
    float iou_thresh);

}  // namespace pellet::imgprocess
