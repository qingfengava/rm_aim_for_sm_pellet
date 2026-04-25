#pragma once

#include <cstdint>
#include <string>

namespace pellet {

struct CameraConfig {
  std::string wust_vl_config_path{"config/camera.yaml"};
  int startup_timeout_ms{1000};
};

struct DetectorConfig {
  int queue_capacity{3};
  int queue_valid_ms{1000};
  int pop_poll_ms{2};
  int detect_pop_timeout_ms{100};
  bool thread_monitor_enable{true};
};

struct MotionConfig {
  int gaussian_ksize{3};
  double gaussian_sigma{0.8};
  bool adaptive_threshold{true};
  int diff_threshold{18};
  int diff_threshold_min{10};
  int diff_threshold_max{35};
  bool morph_enable{false};
  std::string morph_type{"open"};
  int morph_kernel{3};
  int morph_iters{1};
  int area_min{3};
  int area_max{120};
  float ratio_max{4.0F};
  float extent_min{0.2F};
  float contrast_min{0.06F};
  float motion_score_min{0.08F};
  bool nms_enable{true};
  float nms_iou{0.25F};
  int max_candidates{20};
};

struct RoiConfig {
  int output_size{32};
  float size_scale{2.2F};
  int min_crop{20};
  int max_crop{48};
};

struct InferenceConfig {
  std::string backend{"mock"};
  std::string model_path{"model/pellet_cls.onnx"};
  std::string device{"CPU"};
  std::string input_blob_name{"input"};
  std::string output_blob_name{"output"};
  std::string trt_engine_path{"model/pellet_cls.engine"};
  std::string openvino_model_path{"model/pellet_cls.xml"};
  std::string ncnn_param_path{"model/pellet_cls.param"};
  std::string ncnn_bin_path{"model/pellet_cls.bin"};
  int num_threads{1};
  bool use_fp16{false};
  float positive_threshold{0.75F};
  float weak_threshold{0.45F};
  int max_candidates{10};
};

struct DebugConfig {
  bool enable{false};
  int level{0};
  uint32_t modules_mask{0};
};

struct PelletConfig {
  CameraConfig camera{};
  DetectorConfig detector{};
  MotionConfig motion{};
  RoiConfig roi{};
  InferenceConfig inference{};
  DebugConfig debug{};
};

bool LoadConfigFromYaml(const std::string& path, PelletConfig* config);

}  // namespace pellet
