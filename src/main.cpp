#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "pellet/config.h"
#include "pellet/detector/detector.h"
#include "pellet/type.h"

namespace {

volatile std::sig_atomic_t g_running = 1;

void HandleSignal(int /*signal*/) {
  g_running = 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  const std::string config_path = (argc > 1) ? argv[1] : "config/pellet.yaml";

  pellet::PelletConfig config;
  if (!pellet::LoadConfigFromYaml(config_path, &config)) {
    std::cerr << "Failed to read config from " << config_path << ", using defaults.\n";
  }

  pellet::PelletDetector detector(config);
  if (!detector.Start()) {
    std::cerr << "Failed to start detector.\n";
    return 1;
  }

  std::cout << "pellet_detector started with camera source: " << config.camera.source << "\n";
  std::cout << "Collecting results continuously (1s summary). Press Ctrl+C to stop.\n";

  using Clock = std::chrono::steady_clock;
  auto window_begin = Clock::now();
  int window_polls = 0;
  int window_non_empty = 0;
  std::size_t window_detections = 0;

  while (g_running != 0) {
    std::vector<pellet::Detection> detections;
    ++window_polls;
    if (detector.PopDetections(&detections, 100)) {
      ++window_non_empty;
      window_detections += detections.size();
    }

    const auto now = Clock::now();
    if ((now - window_begin) >= std::chrono::seconds(1)) {
      const double avg_det_per_hit =
          (window_non_empty > 0) ? static_cast<double>(window_detections) / window_non_empty : 0.0;
      std::cout << "[1s] polls=" << window_polls
                << ", hit_batches=" << window_non_empty
                << ", detections=" << window_detections
                << ", avg_det_per_hit=" << std::fixed << std::setprecision(2) << avg_det_per_hit << "\n";

      window_begin = now;
      window_polls = 0;
      window_non_empty = 0;
      window_detections = 0;
    }
  }

  if (window_polls > 0) {
    const double avg_det_per_hit =
        (window_non_empty > 0) ? static_cast<double>(window_detections) / window_non_empty : 0.0;
    std::cout << "[last] polls=" << window_polls
              << ", hit_batches=" << window_non_empty
              << ", detections=" << window_detections
              << ", avg_det_per_hit=" << std::fixed << std::setprecision(2) << avg_det_per_hit << "\n";
  }

  std::cout << "Stopping detector...\n";
  detector.Stop();
  return 0;
}
