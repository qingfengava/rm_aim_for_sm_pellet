#include "pellet/imgprocess/preprocess.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

namespace pellet::imgprocess {

// 视频流专用处理器（复用buffer，避免重复内存分配）
class VideoFrameProcessor {
private:
    cv::Mat m_gray_buffer;      // 复用的灰度图buffer
    cv::Mat m_blur_buffer;      // 复用的模糊图buffer
    int m_last_ksize = -1;      // 记录上次的ksize
    
public:
    // 处理单帧（原地复用buffer）
    const cv::Mat& process(cv::Mat& frame, int ksize = 5) {
        if (frame.empty()) {
            return m_blur_buffer;
        }
        
        // 1. 灰度转换（直接使用buffer）
        if (frame.channels() == 3) {
            cv::cvtColor(frame, m_gray_buffer, cv::COLOR_BGR2GRAY);
        } else if (frame.channels() == 1) {
            m_gray_buffer = frame;  // 浅拷贝，不复制数据
        }
        
        // 2. 确保ksize有效
        if (ksize < 1) ksize = 1;
        if (ksize % 2 == 0) ksize++;
        
        // 3. 高斯模糊（如果ksize变化，需要重新分配buffer）
        if (ksize != m_last_ksize) {
            m_blur_buffer.release();  // 释放旧buffer
            m_last_ksize = ksize;
        }
        
        cv::GaussianBlur(m_gray_buffer, m_blur_buffer, 
                        cv::Size(ksize, ksize), 0, 0);
        
        return m_blur_buffer;
    }
    
    // 重置buffer（切换视频源时调用）
    void reset() {
        m_gray_buffer.release();
        m_blur_buffer.release();
        m_last_ksize = -1;
    }
};

// 普通版本（每帧都重新分配内存，适合单张图片）
cv::Mat ToGrayAndBlur(const cv::Mat& frame_bgr, int ksize = 5) {
    if (frame_bgr.empty()) return {};
    
    cv::Mat gray, result;
    if (frame_bgr.channels() == 3) {
        cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame_bgr.clone();  // 每次都会复制数据
    }
    
    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(gray, result, cv::Size(ksize, ksize), 0, 0);
    
    return result;  // 返回新对象，会分配内存
}

}  // namespace pellet::imgprocess
//普通的处理可以调用的原函数，视频流的处理调用类内的函数，内存复用，原地变换，减少内存的开销