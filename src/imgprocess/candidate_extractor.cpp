#include "pellet/imgprocess/candidate_extractor.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {

std::vector<Candidate> ExtractCandidates(
    const cv::Mat& binary_mask,
    const cv::Mat& gray_frame,
    const cv::Mat& motion_response) {
  std::vector<Candidate> candidates;
  if (binary_mask.empty() || gray_frame.empty() || motion_response.empty()) {
    return candidates;
  }

  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  const int num_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);

  candidates.reserve(std::max(0, num_labels - 1));

  for (int label = 1; label < num_labels; ++label) {
    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
    if (area <= 0) {
      continue;
    }

    const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
    const int y = stats.at<int>(label, cv::CC_STAT_TOP);
    const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
    const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    const cv::Rect bbox{x, y, w, h};

    if (bbox.x < 0 || bbox.y < 0 || bbox.br().x > gray_frame.cols || bbox.br().y > gray_frame.rows) {
      continue;
    }

    const double cx = centroids.at<double>(label, 0);
    const double cy = centroids.at<double>(label, 1);

    const cv::Scalar gray_mean = cv::mean(gray_frame(bbox));
    const cv::Scalar motion_mean = cv::mean(motion_response(bbox));

    const float w_f = static_cast<float>(w);
    const float h_f = static_cast<float>(std::max(1, h));
    const float aspect_ratio = std::max(w_f / h_f, h_f / std::max(1.0F, w_f));
    const float circularity_proxy = std::min(w_f, h_f) / std::max(w_f, h_f);

    Candidate candidate;
    candidate.bbox = bbox;
    candidate.center = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
    candidate.area = area;
    candidate.motion_score = static_cast<float>(motion_mean[0] / 255.0);
    candidate.brightness = static_cast<float>(gray_mean[0] / 255.0);
    candidate.circularity = circularity_proxy;
    candidate.aspect_ratio = aspect_ratio;
    candidates.push_back(candidate);
  }

  return candidates;
}

}  // namespace pellet::imgprocess
