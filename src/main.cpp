#include <atomic>
#include <cstdint>
#include <iomanip>
#include <string>
#include <vector>

#include <wust_vl/common/utils/logger.hpp>
#include <wust_vl/common/utils/signal.hpp>
#include <wust_vl/common/utils/timer.hpp>

#include "pellet/config.hpp"
#include "pellet/detector/detector.hpp"
#include "pellet/type.hpp"
#include "pellet/utils/debug_utils.hpp"

int main(int argc, char** argv) {
  const std::string config_path = (argc > 1) ? argv[1] : "config/pellet.yaml";

  // Initialize logger before config loading so early failures use unified logging.
  wust_vl::initLogger("INFO", "logs", true, true, false);

  pellet::PelletConfig config;
  if (!pellet::LoadConfigFromYaml(config_path, &config)) {
    WUST_WARN("main") << "Failed to read config from " << config_path
                      << ", using defaults.";
  }
  const bool enable_runtime_stats =
      pellet::utils::IsDebugEnabled(config, pellet::utils::DebugFeature::kStats1s);
  if (pellet::utils::IsAnyDebugEnabled(config)) {
    __wust_vl_logger::Logger::getInstance().setLevel("DEBUG");
  }

  pellet::PelletDetector detector(config);
  if (!detector.Start()) {
    WUST_ERROR("main") << "Failed to start detector.";
    return 1;
  }

  wust_vl::common::utils::SignalHandler signal_handler;
  signal_handler.start([&detector]() {
    WUST_MAIN("main") << "Stopping detector...";
    detector.Stop();
  });

  WUST_MAIN("main") << "pellet_detector started with camera config: "
                    << config.camera.wust_vl_config_path;
  WUST_MAIN("main") << "Collecting results continuously (1s summary). Press Ctrl+C to stop.";

  std::atomic<int> window_polls{0};
  std::atomic<int> window_non_empty{0};
  std::atomic<std::uint64_t> window_detections{0};

  wust_vl::common::utils::Timer stats_timer("main_stats_timer");
  if (enable_runtime_stats) {
    stats_timer.start(1.0, [&](double dt_ms) {
      const int polls = window_polls.exchange(0, std::memory_order_relaxed);
      const int non_empty = window_non_empty.exchange(0, std::memory_order_relaxed);
      const std::uint64_t detections =
          window_detections.exchange(0, std::memory_order_relaxed);
      const double avg_det_per_hit =
          (non_empty > 0) ? static_cast<double>(detections) / non_empty : 0.0;

      WUST_INFO("main") << "[1s] dt_ms=" << std::fixed << std::setprecision(2) << dt_ms
                        << ", polls=" << polls
                        << ", hit_batches=" << non_empty
                        << ", detections=" << detections
                        << ", avg_det_per_hit=" << std::fixed << std::setprecision(2)
                        << avg_det_per_hit;
    });
  }

  while (!signal_handler.shouldExit()) {
    std::vector<pellet::Detection> detections;
    if (enable_runtime_stats) {
      window_polls.fetch_add(1, std::memory_order_relaxed);
    }
    if (detector.PopDetections(&detections, 100)) {
      if (enable_runtime_stats) {
        window_non_empty.fetch_add(1, std::memory_order_relaxed);
        window_detections.fetch_add(
            static_cast<std::uint64_t>(detections.size()),
            std::memory_order_relaxed);
      }
    }
  }

  if (enable_runtime_stats) {
    stats_timer.stop();
  }

  if (enable_runtime_stats) {
    const int final_polls = window_polls.load(std::memory_order_relaxed);
    const int final_non_empty = window_non_empty.load(std::memory_order_relaxed);
    const std::uint64_t final_detections =
        window_detections.load(std::memory_order_relaxed);
    if (final_polls > 0) {
      const double avg_det_per_hit =
          (final_non_empty > 0)
              ? static_cast<double>(final_detections) / final_non_empty
              : 0.0;
      WUST_INFO("main") << "[last] polls=" << final_polls
                        << ", hit_batches=" << final_non_empty
                        << ", detections=" << final_detections
                        << ", avg_det_per_hit=" << std::fixed << std::setprecision(2)
                        << avg_det_per_hit;
    }
  }

  signal_handler.requestExit();
  return 0;
}
