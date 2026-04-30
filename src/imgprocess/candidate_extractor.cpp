#include "pellet/imgprocess/candidate_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {
namespace {

constexpr double kMinStableWeightSum = 1e-6;
constexpr int kMinBgPixels = 16;

struct LabelAccumulator {
  int area{0};
  double sum_gray{0.0};
  double sum_motion{0.0};
  double sum_w{0.0};
  double sum_wx{0.0};
  double sum_wy{0.0};
};

double ComputeRingBackgroundMean(
    const cv::Mat& gray_u8,
    const cv::Mat& binary_mask_u8,
    const cv::Rect& bbox) {
  if (gray_u8.empty() || binary_mask_u8.empty()) {
    return 0.0;
  }

  const int max_side = std::max(bbox.width, bbox.height);
  const int pad = std::max(2, static_cast<int>(std::round(static_cast<double>(max_side) * 0.5)));

  const cv::Rect frame_rect{0, 0, gray_u8.cols, gray_u8.rows};
  cv::Rect expanded{bbox.x - pad, bbox.y - pad, bbox.width + 2 * pad, bbox.height + 2 * pad};
  expanded &= frame_rect;
  if (expanded.width <= 0 || expanded.height <= 0) {
    return 0.0;
  }

  double sum_bg = 0.0;
  int count_bg = 0;
  const int bbox_x2 = bbox.x + bbox.width;
  const int bbox_y2 = bbox.y + bbox.height;

  for (int y = expanded.y; y < expanded.y + expanded.height; ++y) {
    const std::uint8_t* gray_row = gray_u8.ptr<std::uint8_t>(y);
    const std::uint8_t* mask_row = binary_mask_u8.ptr<std::uint8_t>(y);
    for (int x = expanded.x; x < expanded.x + expanded.width; ++x) {
      const bool in_bbox = (x >= bbox.x) && (x < bbox_x2) && (y >= bbox.y) && (y < bbox_y2);
      if (in_bbox) {
        continue;
      }
      if (mask_row[x] != 0U) {
        continue;
      }

      sum_bg += static_cast<double>(gray_row[x]);
      ++count_bg;
    }
  }

  if (count_bg < kMinBgPixels) {
    return -1.0;
  }
  return sum_bg / static_cast<double>(count_bg);
}

float ComputeRankScore(float motion_score, float local_contrast, float extent, float circularity) {
  return 0.45F * std::clamp(motion_score, 0.0F, 1.0F)
       + 0.35F * std::clamp(local_contrast, 0.0F, 1.0F)
       + 0.10F * std::clamp(extent, 0.0F, 1.0F)
       + 0.10F * std::clamp(circularity, 0.0F, 1.0F);
}

}  // namespace

std::vector<Candidate> ExtractCandidates(
    const cv::Mat& binary_mask,
    const cv::Mat& gray_frame,
    const cv::Mat& motion_response) {
  std::vector<Candidate> candidates;
  if (binary_mask.empty() || gray_frame.empty() || motion_response.empty()) {
    return candidates;
  }

  // Pipeline 保证输入已经是 CV_8UC1，直接使用
  const cv::Mat& gray_u8 = gray_frame;
  const cv::Mat& motion_u8 = motion_response;
  const cv::Mat& mask_u8 = binary_mask;

  if (gray_u8.size() != motion_u8.size() || gray_u8.size() != mask_u8.size()) {
    return candidates;
  }

  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  const int num_labels =
      cv::connectedComponentsWithStats(mask_u8, labels, stats, centroids, 8, CV_32S);
  if (num_labels <= 1) {
    return candidates;
  }

  candidates.reserve(std::max(0, num_labels - 1));
  std::vector<LabelAccumulator> acc(static_cast<std::size_t>(num_labels));

  // 对每个连通域在其 bbox 范围内累积特征，避免扫描全图
  for (int label = 1; label < num_labels; ++label) {
    const int bx = stats.at<int>(label, cv::CC_STAT_LEFT);
    const int by = stats.at<int>(label, cv::CC_STAT_TOP);
    const int bw = stats.at<int>(label, cv::CC_STAT_WIDTH);
    const int bh = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    const cv::Rect bbox{bx, by, bw, bh};

    if (bbox.x < 0 || bbox.y < 0 ||
        bbox.br().x > gray_frame.cols || bbox.br().y > gray_frame.rows) {
      continue;
    }

    LabelAccumulator& a = acc[static_cast<std::size_t>(label)];

    for (int y = by; y < by + bh; ++y) {
      const int* label_row = labels.ptr<int>(y);
      const std::uint8_t* gray_row = gray_u8.ptr<std::uint8_t>(y);
      const std::uint8_t* motion_row = motion_u8.ptr<std::uint8_t>(y);
      for (int x = bx; x < bx + bw; ++x) {
        if (label_row[x] != label) {
          continue;
        }

        const double gray_value = static_cast<double>(gray_row[x]);
        const double motion_value = static_cast<double>(motion_row[x]);
        const double w = motion_value + 1.0;

        ++a.area;
        a.sum_gray += gray_value;
        a.sum_motion += motion_value;
        a.sum_w += w;
        a.sum_wx += w * static_cast<double>(x);
        a.sum_wy += w * static_cast<double>(y);
      }
    }

    const int area = a.area;
    if (area <= 0) {
      continue;
    }

    const int w = bw;
    const int h = bh;

    const double geom_cx = centroids.at<double>(label, 0);
    const double geom_cy = centroids.at<double>(label, 1);
    const double cx = (a.sum_w > kMinStableWeightSum) ? (a.sum_wx / a.sum_w) : geom_cx;
    const double cy = (a.sum_w > kMinStableWeightSum) ? (a.sum_wy / a.sum_w) : geom_cy;
    const double mean_fg = a.sum_gray / static_cast<double>(area);
    const double mean_bg_raw = ComputeRingBackgroundMean(gray_u8, mask_u8, bbox);
    const double mean_bg = (mean_bg_raw >= 0.0) ? mean_bg_raw : mean_fg;
    const float brightness = static_cast<float>(mean_fg / 255.0);
    const float motion_score = static_cast<float>(a.sum_motion / (255.0 * static_cast<double>(area)));
    const float local_contrast = static_cast<float>((mean_fg - mean_bg) / 255.0);

    const float w_f = static_cast<float>(w);
    const float h_f = static_cast<float>(std::max(1, h));
    const float aspect_ratio = std::max(w_f / h_f, h_f / std::max(1.0F, w_f));
    const float circularity_proxy = std::min(w_f, h_f) / std::max(w_f, h_f);
    const float extent = static_cast<float>(
        static_cast<double>(area) / static_cast<double>(std::max(1, w * h)));
    const float rank_score = ComputeRankScore(
        motion_score, local_contrast, extent, circularity_proxy);

    Candidate candidate;
    candidate.bbox = bbox;
    candidate.center = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
    candidate.area = area;
    candidate.motion_score = motion_score;
    candidate.brightness = brightness;
    candidate.circularity = circularity_proxy;
    candidate.aspect_ratio = aspect_ratio;
    candidate.extent = extent;
    candidate.local_contrast = local_contrast;
    candidate.rank_score = rank_score;
    candidates.push_back(candidate);
  }

  return candidates;
}

}  // namespace pellet::imgprocess
