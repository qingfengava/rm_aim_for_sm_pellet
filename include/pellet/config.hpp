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

  // 背景减除
  std::string bg_backend{"knn"};
  int bg_history{30};
  double bg_var_threshold{14.0};
  double bg_learning_rate{-1.0};
  int bg_downsample{2};

  // 全局扰动保护
  double global_fg_ratio_max{0.50};
  double global_response_attenuation{0.25};

  // 形态学
  bool morph_enable{false};
  std::string morph_type{"open"};
  int morph_kernel{3};
  int morph_iters{1};

  // 运动确认
  bool motion_confirm_enable{true};
  int motion_confirm_threshold{10};

  // 候选过滤
  int area_min{3};
  int area_max{150};
  float ratio_max{4.0F};
  float extent_min{0.15F};
  float contrast_min{0.04F};
  float motion_score_min{0.04F};
  bool nms_enable{true};
  float nms_iou{0.25F};
  int max_candidates{6};
};

struct RoiConfig {
  int output_size{32};
  float size_scale{2.2F};
  int min_crop{20};
  int max_crop{48};
};

struct InferenceConfig {
  std::string backend{"onnxruntime"};
  std::string model_path{"model/pellet_cls.onnx"};
  std::string device{"CPU"};
  std::string input_blob_name{"input"};
  std::string output_blob_name{"output"};
  int input_width{32};
  int input_height{32};
  int batch_size{1};
  std::string precision{"fp32"};
  std::string int8_model_path{""};
  std::string engine_path{""};
  bool trt_require_prebuilt_engine{false};
  std::string openvino_model_path{"model/pellet_cls.xml"};
  std::string ncnn_param_path{"model/pellet_cls.param"};
  std::string ncnn_bin_path{"model/pellet_cls.bin"};
  int num_threads{1};
  float positive_threshold{0.75F};
  float weak_threshold{0.45F};
  int max_candidates{2};
};

struct DebugConfig {
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
