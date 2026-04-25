#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <wust_vl/common/utils/logger.hpp>
#include <wust_vl/common/utils/recorder.hpp>
#include <wust_vl/common/utils/signal.hpp>
#include <wust_vl/common/utils/timer.hpp>
#include <wust_vl/video/camera.hpp>
#include <yaml-cpp/yaml.h>

namespace {

struct Options {
  std::string camera_config_path{"config/camera.yaml"};
  std::string output_path{"logs/recorder/frames.rec"};
  int duration_sec{0};   // 0 means unlimited
  int max_frames{0};     // 0 means unlimited
  bool verbose{false};
  bool show_help{false};
};

struct RecordedFrame {
  int64_t timestamp_ms{0};
  uint8_t pixel_format{0};
  int32_t rows{0};
  int32_t cols{0};
  int32_t type{0};
  std::vector<uint8_t> data{};
};

template <typename T>
void WriteScalar(std::ostream& os, const T& value) {
  os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

class RecordedFrameWriter final : public wust_vl::common::utils::Writer<RecordedFrame> {
 public:
  void write(std::ostream& os, const RecordedFrame& frame) override {
    if (!header_written_) {
      WriteFileHeader(os);
      header_written_ = true;
    }

    WriteScalar<uint32_t>(os, kFrameMagic);
    WriteScalar<int64_t>(os, frame.timestamp_ms);
    WriteScalar<uint8_t>(os, frame.pixel_format);
    WriteScalar<int32_t>(os, frame.rows);
    WriteScalar<int32_t>(os, frame.cols);
    WriteScalar<int32_t>(os, frame.type);
    const uint32_t payload_size = static_cast<uint32_t>(frame.data.size());
    WriteScalar<uint32_t>(os, payload_size);
    if (payload_size > 0) {
      os.write(reinterpret_cast<const char*>(frame.data.data()), payload_size);
    }
  }

 private:
  void WriteFileHeader(std::ostream& os) {
    WriteScalar<uint32_t>(os, kFileMagic);
    WriteScalar<uint16_t>(os, kVersionMajor);
    WriteScalar<uint16_t>(os, kVersionMinor);
  }

  static constexpr uint32_t kFileMagic = 0x31434552U;   // "REC1"
  static constexpr uint16_t kVersionMajor = 1;
  static constexpr uint16_t kVersionMinor = 0;
  static constexpr uint32_t kFrameMagic = 0x314D5246U;  // "FRM1"

  bool header_written_{false};
};

template <typename T>
T ReadYamlOr(const YAML::Node& node, const char* key, const T& default_value) {
  const YAML::Node child = node[key];
  if (!child || child.IsNull() || !child.IsScalar()) {
    return default_value;
  }
  try {
    return child.as<T>();
  } catch (const YAML::BadConversion&) {
    return default_value;
  }
}

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

int64_t ToSteadyMs(const std::chrono::steady_clock::time_point& tp) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "Options:\n"
      << "  --camera-config <path>   Camera config yaml (default: config/camera.yaml)\n"
      << "  --output <path>          Record output path (default: logs/recorder/frames.rec)\n"
      << "  --duration-sec <sec>     Max running seconds, 0 = unlimited (default: 0)\n"
      << "  --max-frames <n>         Max recorded frames, 0 = unlimited (default: 0)\n"
      << "  --verbose                Print per-frame debug logs\n"
      << "  --help                   Show this help message\n";
}

bool ParseArgs(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        WUST_ERROR("recorder") << "missing value for " << flag;
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      options->show_help = true;
      return true;
    }
    if (arg == "--camera-config") {
      const char* value = require_value("--camera-config");
      if (value == nullptr) {
        return false;
      }
      options->camera_config_path = value;
      continue;
    }
    if (arg == "--output") {
      const char* value = require_value("--output");
      if (value == nullptr) {
        return false;
      }
      options->output_path = value;
      continue;
    }
    if (arg == "--duration-sec") {
      const char* value = require_value("--duration-sec");
      if (value == nullptr) {
        return false;
      }
      options->duration_sec = std::max(0, std::stoi(value));
      continue;
    }
    if (arg == "--max-frames") {
      const char* value = require_value("--max-frames");
      if (value == nullptr) {
        return false;
      }
      options->max_frames = std::max(0, std::stoi(value));
      continue;
    }
    if (arg == "--verbose") {
      options->verbose = true;
      continue;
    }

    WUST_ERROR("recorder") << "unknown argument: " << arg;
    PrintUsage(argv[0]);
    return false;
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  wust_vl::initLogger("INFO", "logs", true, true, false);

  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    return 1;
  }
  if (options.show_help) {
    return 0;
  }

  YAML::Node camera_yaml;
  try {
    camera_yaml = YAML::LoadFile(options.camera_config_path);
  } catch (const std::exception& e) {
    WUST_ERROR("recorder")
        << "failed to load camera config: " << options.camera_config_path
        << ", err=" << e.what();
    return 1;
  }

  auto writer = std::make_shared<RecordedFrameWriter>();
  wust_vl::common::utils::Recorder<RecordedFrame> recorder(options.output_path, writer);
  try {
    recorder.start();
  } catch (const std::exception& e) {
    WUST_ERROR("recorder") << "failed to start recorder: " << e.what();
    return 1;
  }

  wust_vl::video::Camera camera;
  if (!camera.init(camera_yaml)) {
    WUST_ERROR("recorder") << "camera init failed";
    recorder.stop();
    return 1;
  }

  const YAML::Node hik_camera = camera_yaml["hik_camera"];
  const std::string trigger_type_yaml =
      ReadYamlOr<std::string>(hik_camera, "trigger_type", std::string("none"));
  const std::string trigger_mode_yaml =
      ReadYamlOr<std::string>(hik_camera, "trigger_mode", trigger_type_yaml);
  const bool software_trigger_mode = (ToLower(trigger_mode_yaml) == "software");
  const int software_trigger_interval_ms =
      std::max(0, ReadYamlOr<int>(hik_camera, "software_trigger_interval_ms", 0));
  const std::string sn =
      ReadYamlOr<std::string>(hik_camera, "target_sn", std::string("unknown"));

  std::atomic<bool> should_exit{false};
  std::atomic<bool> max_frames_reached{false};
  std::atomic<uint64_t> frame_count{0};
  std::atomic<uint64_t> byte_count{0};
  std::atomic<uint64_t> drop_count{0};
  std::atomic<uint64_t> window_frames{0};
  std::atomic<uint64_t> window_bytes{0};

  const auto start_tp = std::chrono::steady_clock::now();

  auto on_frame = [&](wust_vl::video::ImageFrame& frame) {
    if (should_exit.load(std::memory_order_relaxed)) {
      return;
    }
    if (frame.src_img.empty()) {
      drop_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    RecordedFrame record;
    record.timestamp_ms = ToSteadyMs(frame.timestamp);
    record.pixel_format = static_cast<uint8_t>(frame.pixel_format);
    record.rows = frame.src_img.rows;
    record.cols = frame.src_img.cols;
    record.type = frame.src_img.type();

    cv::Mat src = frame.src_img;
    if (!src.isContinuous()) {
      src = src.clone();
    }
    const std::size_t bytes = src.total() * src.elemSize();
    if (bytes == 0) {
      drop_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    record.data.resize(bytes);
    std::memcpy(record.data.data(), src.data, bytes);

    recorder.push(record);

    const uint64_t total = frame_count.fetch_add(1, std::memory_order_relaxed) + 1;
    byte_count.fetch_add(bytes, std::memory_order_relaxed);
    window_frames.fetch_add(1, std::memory_order_relaxed);
    window_bytes.fetch_add(bytes, std::memory_order_relaxed);

    if (options.max_frames > 0 &&
        total >= static_cast<uint64_t>(options.max_frames)) {
      max_frames_reached.store(true, std::memory_order_relaxed);
    }

    if (options.verbose) {
      WUST_INFO("recorder")
          << "frame#" << total << " ts=" << record.timestamp_ms
          << " size=" << record.cols << "x" << record.rows
          << " bytes=" << bytes;
    }
  };

  camera.setFrameCallback(on_frame);
  camera.start();

  std::thread software_pull_thread;
  if (software_trigger_mode) {
    WUST_WARN("recorder")
        << "software trigger mode detected, enable active readImage loop";
    software_pull_thread = std::thread([&]() {
      const auto min_interval = std::chrono::milliseconds(software_trigger_interval_ms);
      auto last_pull_tp = std::chrono::steady_clock::time_point{};

      while (!should_exit.load(std::memory_order_relaxed) &&
             !max_frames_reached.load(std::memory_order_relaxed)) {
        if (min_interval.count() > 0) {
          const auto now = std::chrono::steady_clock::now();
          if (last_pull_tp.time_since_epoch().count() != 0) {
            const auto next_tp = last_pull_tp + min_interval;
            if (now < next_tp) {
              std::this_thread::sleep_for(next_tp - now);
            }
          }
          last_pull_tp = std::chrono::steady_clock::now();
        }

        wust_vl::video::ImageFrame frame = camera.readImage();
        if (frame.src_img.empty()) {
          continue;
        }
        on_frame(frame);
      }
    });
  }

  wust_vl::common::utils::SignalHandler signal_handler;
  signal_handler.start([&]() {
    should_exit.store(true, std::memory_order_relaxed);
  });

  wust_vl::common::utils::Timer stats_timer("recorder_stats_timer");
  stats_timer.start(1.0, [&](double dt_ms) {
    const uint64_t frames = window_frames.exchange(0, std::memory_order_relaxed);
    const uint64_t bytes = window_bytes.exchange(0, std::memory_order_relaxed);
    const double fps = (dt_ms > 1e-3) ? (frames * 1000.0 / dt_ms) : 0.0;
    const double mbps = (dt_ms > 1e-3) ? (bytes * 1000.0 / dt_ms / (1024.0 * 1024.0)) : 0.0;

    WUST_INFO("recorder")
        << "[1s] dt_ms=" << std::fixed << std::setprecision(2) << dt_ms
        << ", frames=" << frames
        << ", fps=" << std::fixed << std::setprecision(2) << fps
        << ", throughput=" << std::fixed << std::setprecision(2) << mbps << " MiB/s"
        << ", total_frames=" << frame_count.load(std::memory_order_relaxed)
        << ", dropped=" << drop_count.load(std::memory_order_relaxed);
  });

  WUST_MAIN("recorder")
      << "recorder started, sn=" << sn
      << ", camera_config=" << options.camera_config_path
      << ", output=" << options.output_path
      << ", trigger_mode=" << trigger_mode_yaml
      << ", max_frames=" << options.max_frames
      << ", duration_sec=" << options.duration_sec;

  while (!signal_handler.shouldExit() &&
         !max_frames_reached.load(std::memory_order_relaxed)) {
    if (options.duration_sec > 0) {
      const auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start_tp).count();
      if (elapsed_sec >= options.duration_sec) {
        WUST_MAIN("recorder") << "duration reached: " << options.duration_sec << " s";
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  should_exit.store(true, std::memory_order_relaxed);
  signal_handler.requestExit();
  stats_timer.stop();

  if (software_pull_thread.joinable()) {
    software_pull_thread.join();
  }

  camera.stop();
  camera.setFrameCallback({});
  recorder.stop();

  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_tp).count();
  WUST_MAIN("recorder")
      << "recorder stopped, elapsed_ms=" << elapsed_ms
      << ", frames=" << frame_count.load(std::memory_order_relaxed)
      << ", bytes=" << byte_count.load(std::memory_order_relaxed)
      << ", dropped=" << drop_count.load(std::memory_order_relaxed)
      << ", output=" << std::filesystem::path(options.output_path).lexically_normal().string();
  return 0;
}
