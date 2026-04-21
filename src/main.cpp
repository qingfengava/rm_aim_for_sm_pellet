#include <iostream>
#include <string>
#include <vector>

#include "pellet/config.h"
#include "pellet/detector/detector.h"
#include "pellet/type.h"

int main(int argc, char** argv) {
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
  std::cout << "Collecting results for 200 polling cycles...\n";

  for (int i = 0; i < 200; ++i) {
    std::vector<pellet::Detection> detections;
    if (detector.PopDetections(&detections, 100)) {
      std::cout << "detections: " << detections.size() << "\n";
    }
  }

  detector.Stop();
  return 0;
}
