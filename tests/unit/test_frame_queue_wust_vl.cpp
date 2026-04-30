#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <thread>

#include <opencv2/core.hpp>

#include "pellet/detector/frame_queue.hpp"

namespace {

pellet::detector::FramePacket MakePacket(uint32_t frame_id) {
  pellet::detector::FramePacket packet;
  packet.frame_id = frame_id;
  packet.timestamp_ms = static_cast<int64_t>(frame_id);
  packet.frame_bgr = cv::Mat::zeros(8, 8, CV_8UC3);
  return packet;
}

}  // namespace

TEST(FrameQueueWustVlTest, PopTimeoutKeepsOriginalSemantics) {
  pellet::detector::FrameQueue queue(/*capacity=*/3, /*queue_valid_ms=*/1000, /*pop_poll_ms=*/2);

  pellet::detector::FramePacket out;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = queue.Pop(&out, /*timeout_ms=*/40);
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - t0)
                              .count();

  EXPECT_FALSE(ok);
  EXPECT_GE(elapsed_ms, 30);
}

TEST(FrameQueueWustVlTest, StopUnblocksPendingPop) {
  pellet::detector::FrameQueue queue(/*capacity=*/3, /*queue_valid_ms=*/1000, /*pop_poll_ms=*/2);

  auto fut = std::async(std::launch::async, [&queue]() {
    pellet::detector::FramePacket out;
    return queue.Pop(&out, /*timeout_ms=*/5000);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  queue.Stop();

  const auto status = fut.wait_for(std::chrono::milliseconds(300));
  ASSERT_EQ(status, std::future_status::ready);
  EXPECT_FALSE(fut.get());
}

TEST(FrameQueueWustVlTest, DiscardsStaleFramesByValidWindow) {
  pellet::detector::FrameQueue queue(/*capacity=*/3, /*queue_valid_ms=*/30, /*pop_poll_ms=*/2);

  queue.Push(MakePacket(1));
  std::this_thread::sleep_for(std::chrono::milliseconds(60));

  pellet::detector::FramePacket out;
  EXPECT_FALSE(queue.Pop(&out, /*timeout_ms=*/10));

  queue.Push(MakePacket(2));
  EXPECT_TRUE(queue.Pop(&out, /*timeout_ms=*/30));
  EXPECT_EQ(out.frame_id, 2U);
}

TEST(FrameQueueWustVlTest, CapacityOverflowDropsOldestKeepsNewest) {
  pellet::detector::FrameQueue queue(/*capacity=*/2, /*queue_valid_ms=*/1000, /*pop_poll_ms=*/2);

  queue.Push(MakePacket(1));
  queue.Push(MakePacket(2));
  queue.Push(MakePacket(3));

  pellet::detector::FramePacket out;
  ASSERT_TRUE(queue.Pop(&out, /*timeout_ms=*/20));
  EXPECT_EQ(out.frame_id, 2U);
  ASSERT_TRUE(queue.Pop(&out, /*timeout_ms=*/20));
  EXPECT_EQ(out.frame_id, 3U);
  EXPECT_FALSE(queue.Pop(&out, /*timeout_ms=*/10));

  const auto stats = queue.GetStatsSnapshot();
  EXPECT_EQ(stats.capacity, 2U);
  EXPECT_EQ(stats.push_total, 3U);
  EXPECT_EQ(stats.pop_total, 2U);
  EXPECT_EQ(stats.drop_overflow, 1U);
  EXPECT_EQ(stats.drop_stale, 0U);
}

TEST(FrameQueueWustVlTest, SnapshotTracksStaleDrops) {
  pellet::detector::FrameQueue queue(/*capacity=*/3, /*queue_valid_ms=*/20, /*pop_poll_ms=*/2);

  queue.Push(MakePacket(1));
  queue.Push(MakePacket(2));
  std::this_thread::sleep_for(std::chrono::milliseconds(40));

  pellet::detector::FramePacket out;
  EXPECT_FALSE(queue.Pop(&out, /*timeout_ms=*/10));
  const auto stats = queue.GetStatsSnapshot();
  EXPECT_EQ(stats.push_total, 2U);
  EXPECT_EQ(stats.pop_total, 0U);
  EXPECT_GE(stats.drop_stale, 1U);
}
