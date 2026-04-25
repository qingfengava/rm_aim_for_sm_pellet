#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <wust_vl/common/concurrency/monitored_thread.hpp>

#include "pellet/config.hpp"
#include "pellet/detector/detect_worker.hpp"
#include "pellet/detector/detector_pipeline.hpp"
#include "pellet/detector/frame_queue.hpp"
#include "pellet/infer/i_classifier.hpp"

namespace {

class AlwaysPositiveClassifier final : public pellet::infer::IClassifier {
 public:
  bool Init(const pellet::InferenceConfig&) override {
    return true;
  }

  std::vector<float> Infer(const std::vector<cv::Mat>& rois) override {
    return std::vector<float>(rois.size(), 1.0F);
  }
};

pellet::detector::FramePacket MakePacket(uint32_t frame_id, const cv::Mat& frame_bgr) {
  pellet::detector::FramePacket packet;
  packet.frame_id = frame_id;
  packet.timestamp_ms = static_cast<int64_t>(frame_id);
  packet.frame_bgr = frame_bgr.clone();
  return packet;
}

cv::Mat MakeFrameWithTarget(bool has_target) {
  cv::Mat frame = cv::Mat::zeros(64, 64, CV_8UC3);
  if (has_target) {
    cv::rectangle(frame, cv::Rect(28, 28, 6, 6), cv::Scalar(255, 255, 255), cv::FILLED);
  }
  return frame;
}

pellet::PelletConfig MakeLenientConfig() {
  pellet::PelletConfig config;
  config.motion.gaussian_ksize = 1;
  config.motion.gaussian_sigma = 0.0;
  config.motion.diff_threshold = 5;
  config.motion.diff_threshold_min = 3;
  config.motion.diff_threshold_max = 8;
  config.motion.morph_enable = false;
  config.motion.area_min = 1;
  config.motion.area_max = 2048;
  config.motion.ratio_max = 10.0F;
  config.motion.extent_min = 0.0F;
  config.motion.contrast_min = -1.0F;
  config.motion.motion_score_min = 0.0F;
  config.motion.nms_enable = false;
  config.motion.max_candidates = 20;
  config.inference.positive_threshold = 0.5F;
  config.inference.max_candidates = 10;
  return config;
}

}  // namespace

TEST(DetectWorkerThreadingTest, StartStopKeepsHeartbeatAliveWithoutFrames) {
  auto classifier = std::make_shared<AlwaysPositiveClassifier>();
  ASSERT_TRUE(classifier->Init(pellet::InferenceConfig{}));

  pellet::PelletConfig config = MakeLenientConfig();
  pellet::detector::DetectorPipeline pipeline(config, classifier);
  pellet::detector::FrameQueue frame_queue(/*capacity=*/3, /*queue_valid_ms=*/1000, /*pop_poll_ms=*/2);
  pellet::detector::DetectWorker worker(
      &frame_queue, &pipeline, /*frame_pop_timeout_ms=*/100, /*thread_monitor_enable=*/true,
      /*show_thread_status=*/false);

  auto& manager = wust_vl::common::concurrency::ThreadManager::instance();
  manager.unregisterThread("detector_worker");

  worker.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(2300));

  const auto statuses = manager.getAllThreadStatuses();
  const auto it = statuses.find("detector_worker");
  ASSERT_NE(it, statuses.end());
  EXPECT_NE(it->second, wust_vl::common::concurrency::MonitoredThread::Status::Hung);

  frame_queue.Stop();
  worker.Stop();

  const auto statuses_after = manager.getAllThreadStatuses();
  EXPECT_EQ(statuses_after.find("detector_worker"), statuses_after.end());
}

TEST(DetectWorkerThreadingTest, ProducesDetectionsWhenFramesArrive) {
  auto classifier = std::make_shared<AlwaysPositiveClassifier>();
  ASSERT_TRUE(classifier->Init(pellet::InferenceConfig{}));

  pellet::PelletConfig config = MakeLenientConfig();
  pellet::detector::DetectorPipeline pipeline(config, classifier);
  pellet::detector::FrameQueue frame_queue(/*capacity=*/8, /*queue_valid_ms=*/1000, /*pop_poll_ms=*/1);
  pellet::detector::DetectWorker worker(
      &frame_queue, &pipeline, /*frame_pop_timeout_ms=*/100, /*thread_monitor_enable=*/true,
      /*show_thread_status=*/false);

  worker.Start();

  frame_queue.Push(MakePacket(0, MakeFrameWithTarget(false)));
  frame_queue.Push(MakePacket(1, MakeFrameWithTarget(true)));
  frame_queue.Push(MakePacket(2, MakeFrameWithTarget(true)));

  std::this_thread::sleep_for(std::chrono::milliseconds(120));

  std::vector<pellet::Detection> detections;
  const bool ok = worker.PopLatest(&detections, /*timeout_ms=*/1500);

  frame_queue.Stop();
  worker.Stop();

  ASSERT_TRUE(ok);
  ASSERT_FALSE(detections.empty());
  EXPECT_EQ(detections.front().frame_id, 2U);
}
