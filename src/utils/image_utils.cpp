#include "pellet/utils/image_utils.hpp"

#include <algorithm>

namespace pellet::utils {

cv::Rect ClampRect(const cv::Rect& rect, const cv::Size& bounds) {
  const int x = std::clamp(rect.x, 0, bounds.width);
  const int y = std::clamp(rect.y, 0, bounds.height);
  const int max_width = std::max(0, bounds.width - x);
  const int max_height = std::max(0, bounds.height - y);
  const int width = std::clamp(rect.width, 0, max_width);
  const int height = std::clamp(rect.height, 0, max_height);
  return {x, y, width, height};
}

cv::Rect MakeSquareRoi(const cv::Point2f& center, int side, const cv::Size& bounds) {
  const int half = std::max(1, side / 2);
  const int x = static_cast<int>(center.x) - half;
  const int y = static_cast<int>(center.y) - half;
  return ClampRect(cv::Rect{x, y, side, side}, bounds);
}

}  // namespace pellet::utils
