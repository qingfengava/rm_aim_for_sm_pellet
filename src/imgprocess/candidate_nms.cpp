#include "pellet/imgprocess/candidate_nms.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace pellet::imgprocess {
namespace {

  //计算两个框的重叠程度
float ComputeIou(const cv::Rect& a, const cv::Rect& b) {
  const int ax2 = a.x + a.width;
  const int ay2 = a.y + a.height;
  const int bx2 = b.x + b.width;
  const int by2 = b.y + b.height;

  const int ix1 = std::max(a.x, b.x);
  const int iy1 = std::max(a.y, b.y);
  const int ix2 = std::min(ax2, bx2);
  const int iy2 = std::min(ay2, by2);

  const int inter_w = std::max(0, ix2 - ix1);
  const int inter_h = std::max(0, iy2 - iy1);
  if (inter_w <= 0 || inter_h <= 0) {
    return 0.0F;
  }

  const float inter_area = static_cast<float>(inter_w * inter_h);
  const float area_a = static_cast<float>(std::max(0, a.width) * std::max(0, a.height));
  const float area_b = static_cast<float>(std::max(0, b.width) * std::max(0, b.height));
  const float union_area = area_a + area_b - inter_area;
  if (union_area <= 0.0F) {
    return 0.0F;
  }

  return inter_area / union_area;
}

}  // namespace

//基于IoU的非极大值抑制
std::vector<Candidate> ApplyNms(
    const std::vector<Candidate>& candidates,
    float iou_thresh) {
  if (candidates.empty()) {
    return {};
  }

  const float iou_threshold = std::clamp(iou_thresh, 0.0F, 1.0F);

  std::vector<int> order(candidates.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&candidates](int lhs, int rhs) {
    const Candidate& a = candidates[static_cast<std::size_t>(lhs)];
    const Candidate& b = candidates[static_cast<std::size_t>(rhs)];
    if (std::fabs(a.rank_score - b.rank_score) > 1e-6F) {
      return a.rank_score > b.rank_score;
    }
    if (a.area != b.area) {
      return a.area > b.area;
    }
    return a.brightness > b.brightness;
  });

  std::vector<bool> suppressed(candidates.size(), false);
  std::vector<Candidate> kept;
  kept.reserve(candidates.size());

  for (std::size_t i = 0; i < order.size(); ++i) {
    const int current_idx = order[i];
    const std::size_t current_u = static_cast<std::size_t>(current_idx);
    if (suppressed[current_u]) {
      continue;
    }

    const Candidate& current = candidates[current_u];
    kept.push_back(current);

    for (std::size_t j = i + 1; j < order.size(); ++j) {
      const int other_idx = order[j];
      const std::size_t other_u = static_cast<std::size_t>(other_idx);
      if (suppressed[other_u]) {
        continue;
      }

      const float iou = ComputeIou(current.bbox, candidates[other_u].bbox);
      if (iou > iou_threshold) {
        suppressed[other_u] = true;
      }
    }
  }

  return kept;
}

}  // namespace pellet::imgprocess
