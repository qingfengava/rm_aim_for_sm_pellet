#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>

#include "pellet/imgprocess/candidate_extractor.hpp"

namespace pellet::imgprocess {

std::vector<Candidate> FilterByMotionConfirm(
    const std::vector<Candidate>& candidates,
    const cv::Mat& motion_response,
    int threshold);

}  // namespace pellet::imgprocess
