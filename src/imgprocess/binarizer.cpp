#include "pellet/imgprocess/binarizer.h"

#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {

class MotionBinarizer {
public:
    MotionBinarizer() = default;
    
    // 主处理函数，支持内存复用
    cv::Mat Process(const cv::Mat& motion_response, int threshold_low, int threshold_high) {
        if (motion_response.empty()) {
            return {};
        }
        
        // 确保工作矩阵大小和类型匹配
        EnsureMatrices(motion_response.size(), motion_response.type());
        
        // 弱阈值二值化
        cv::threshold(motion_response, weak_, threshold_low, 255, cv::THRESH_BINARY);
        
        if (threshold_high <= threshold_low) {
            return weak_.clone();
        }
        
        // 强阈值二值化
        cv::threshold(motion_response, strong_, threshold_high, 255, cv::THRESH_BINARY);
        
        // 膨胀强二值化结果
        cv::dilate(strong_, strong_dilated_, 
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size{3, 3}));
        
        // 连接弱响应和强响应区域
        cv::bitwise_and(weak_, strong_dilated_, linked_);
        
        // 合并结果
        cv::bitwise_or(linked_, strong_, result_);
        
        return result_;
    }
    
    // 获取内部矩阵（用于调试或进一步处理）
    const cv::Mat& GetWeak() const { return weak_; }
    const cv::Mat& GetStrong() const { return strong_; }
    const cv::Mat& GetLinked() const { return linked_; }
    
    // 释放内部缓存
    void Release() {
        weak_.release();
        strong_.release();
        strong_dilated_.release();
        linked_.release();
        result_.release();
    }

private:
    void EnsureMatrices(cv::Size size, int type) {
        // 检查每个矩阵是否需要重新分配
        if (weak_.size() != size || weak_.type() != type) {
            weak_ = cv::Mat(size, type);
            strong_ = cv::Mat(size, type);
            strong_dilated_ = cv::Mat(size, type);
            linked_ = cv::Mat(size, type);
            result_ = cv::Mat(size, type);
        }
        
        // 确保矩阵为零（可选，根据需求）
        // weak_.setTo(0);
        // strong_.setTo(0);
        // strong_dilated_.setTo(0);
        // linked_.setTo(0);
        // result_.setTo(0);
    }
    
    cv::Mat weak_;
    cv::Mat strong_;
    cv::Mat strong_dilated_;
    cv::Mat linked_;
    cv::Mat result_;
};

// 全局单例实例（用于保持原函数接口）
static MotionBinarizer& GetGlobalBinarizer() {
    static MotionBinarizer instance;
    return instance;
}

// 保持原有函数接口不变
cv::Mat BinarizeMotion(const cv::Mat& motion_response, int threshold_low, int threshold_high) {
    return GetGlobalBinarizer().Process(motion_response, threshold_low, threshold_high);
}

// 可选：提供带实例参数的版本，支持多实例和更好的内存复用控制
cv::Mat BinarizeMotion(MotionBinarizer& binarizer, const cv::Mat& motion_response, 
                       int threshold_low, int threshold_high) {
    return binarizer.Process(motion_response, threshold_low, threshold_high);
}

// 辅助函数：创建新的二值化器实例
std::unique_ptr<MotionBinarizer> CreateMotionBinarizer() {
    return std::make_unique<MotionBinarizer>();
}

}  // namespace pellet::imgprocess