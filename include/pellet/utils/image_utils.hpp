#pragma once

#include <opencv2/core/types.hpp>

namespace pellet::utils {

cv::Rect ClampRect(const cv::Rect& rect, const cv::Size& bounds);
cv::Rect MakeSquareRoi(const cv::Point2f& center, int side, const cv::Size& bounds);

}  // namespace pellet::utils
