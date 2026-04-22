#include "pellet/imgprocess/preprocess.h"
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {

// 灰度处理
cv::Mat togray(const cv::Mat& frame) {
    if (frame.empty()) {
        return {};
    }
    cv::Mat al_gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, al_gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 1) {
        al_gray = frame.clone();  // 已经是灰度图，直接克隆
    } else {
        return {};  // 不支持的通道数
    }
    return al_gray;
}

// 高斯去噪
cv::Mat togaussian(const cv::Mat& al_gray, int ksize) {
    if (al_gray.empty()) {
        return {};
    }
    cv::Mat al_guassin;
    cv::GaussianBlur(al_gray, al_guassin, cv::Size(ksize, ksize), 0, 0);
    return al_guassin;
}

// 合并处理：灰度 + 高斯模糊
cv::Mat ToGrayAndBlur(const cv::Mat& frame_bgr, int ksize) {
    if (frame_bgr.empty()) {
        return {};
    }
    
    // 1. 转换为灰度
    cv::Mat gray = togray(frame_bgr);
    if (gray.empty()) {
        return {};
    }
    
    // 2. 确保ksize为奇数且不小于1
    if (ksize < 1) {
        ksize = 1;
    }
    if (ksize % 2 == 0) {
        ++ksize;
    }
    
    // 3. 高斯模糊
    cv::Mat blurred = togaussian(gray, ksize);
    return blurred;
}

}  // namespace pellet::imgprocess