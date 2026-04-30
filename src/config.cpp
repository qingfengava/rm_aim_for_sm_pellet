#include "pellet/config.hpp"

#include <opencv2/core.hpp>

namespace {

template <typename T>
void ReadIfPresent(const cv::FileNode& node, const char* key, T* value) {
  const cv::FileNode child = node[key];
  if (!child.empty()) {
    child >> *value;
  }
}

void ReadBoolIfPresent(const cv::FileNode& node, const char* key, bool* value) {
  const cv::FileNode child = node[key];
  if (!child.empty()) {
    int raw = 0;
    child >> raw;
    *value = (raw != 0);
  }
}

void ReadUInt32IfPresent(const cv::FileNode& node, const char* key, uint32_t* value) {
  const cv::FileNode child = node[key];
  if (child.empty()) {
    return;
  }
  int raw = 0;
  child >> raw;
  if (raw <= 0) {
    *value = 0U;
    return;
  }
  *value = static_cast<uint32_t>(raw);
}

}  // namespace

namespace pellet {

bool LoadConfigFromYaml(const std::string& path, PelletConfig* config) {
  if (config == nullptr) {
    return false;
  }

  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  const cv::FileNode camera = fs["camera"];
  if (!camera.empty()) {
    ReadIfPresent(camera, "wust_vl_config_path", &config->camera.wust_vl_config_path);
    ReadIfPresent(camera, "startup_timeout_ms", &config->camera.startup_timeout_ms);
  }

  const cv::FileNode detector = fs["detector"];
  if (!detector.empty()) {
    ReadIfPresent(detector, "queue_capacity", &config->detector.queue_capacity);
    ReadIfPresent(detector, "queue_valid_ms", &config->detector.queue_valid_ms);
    ReadIfPresent(detector, "pop_poll_ms", &config->detector.pop_poll_ms);
    ReadIfPresent(detector, "detect_pop_timeout_ms", &config->detector.detect_pop_timeout_ms);
    ReadBoolIfPresent(
        detector, "thread_monitor_enable", &config->detector.thread_monitor_enable);
  }

  const cv::FileNode motion = fs["motion"];
  if (!motion.empty()) {
    ReadIfPresent(motion, "gaussian_ksize", &config->motion.gaussian_ksize);
    ReadIfPresent(motion, "gaussian_sigma", &config->motion.gaussian_sigma);

    ReadIfPresent(motion, "bg_backend", &config->motion.bg_backend);
    ReadIfPresent(motion, "bg_history", &config->motion.bg_history);
    ReadIfPresent(motion, "bg_var_threshold", &config->motion.bg_var_threshold);
    ReadIfPresent(motion, "bg_learning_rate", &config->motion.bg_learning_rate);
    ReadIfPresent(motion, "bg_downsample", &config->motion.bg_downsample);

    ReadIfPresent(motion, "global_fg_ratio_max", &config->motion.global_fg_ratio_max);
    ReadIfPresent(motion, "global_response_attenuation", &config->motion.global_response_attenuation);

    ReadBoolIfPresent(motion, "morph_enable", &config->motion.morph_enable);
    ReadIfPresent(motion, "morph_type", &config->motion.morph_type);
    ReadIfPresent(motion, "morph_kernel", &config->motion.morph_kernel);
    ReadIfPresent(motion, "morph_iters", &config->motion.morph_iters);

    ReadBoolIfPresent(motion, "motion_confirm_enable", &config->motion.motion_confirm_enable);
    ReadIfPresent(motion, "motion_confirm_threshold", &config->motion.motion_confirm_threshold);

    ReadIfPresent(motion, "area_min", &config->motion.area_min);
    ReadIfPresent(motion, "area_max", &config->motion.area_max);
    ReadIfPresent(motion, "ratio_max", &config->motion.ratio_max);
    ReadIfPresent(motion, "extent_min", &config->motion.extent_min);
    ReadIfPresent(motion, "contrast_min", &config->motion.contrast_min);
    ReadIfPresent(motion, "motion_score_min", &config->motion.motion_score_min);
    ReadBoolIfPresent(motion, "nms_enable", &config->motion.nms_enable);
    ReadIfPresent(motion, "nms_iou", &config->motion.nms_iou);
    ReadIfPresent(motion, "max_candidates", &config->motion.max_candidates);
  }

  const cv::FileNode roi = fs["roi"];
  if (!roi.empty()) {
    ReadIfPresent(roi, "output_size", &config->roi.output_size);
    ReadIfPresent(roi, "size_scale", &config->roi.size_scale);
    ReadIfPresent(roi, "min_crop", &config->roi.min_crop);
    ReadIfPresent(roi, "max_crop", &config->roi.max_crop);
  }

  const cv::FileNode inference = fs["inference"];
  if (!inference.empty()) {
    ReadIfPresent(inference, "backend", &config->inference.backend);
    ReadIfPresent(inference, "model_path", &config->inference.model_path);
    ReadIfPresent(inference, "device", &config->inference.device);
    ReadIfPresent(inference, "input_blob_name", &config->inference.input_blob_name);
    ReadIfPresent(inference, "output_blob_name", &config->inference.output_blob_name);
    ReadIfPresent(inference, "input_width", &config->inference.input_width);
    ReadIfPresent(inference, "input_height", &config->inference.input_height);
    ReadIfPresent(inference, "batch_size", &config->inference.batch_size);
    ReadIfPresent(inference, "precision", &config->inference.precision);
    ReadIfPresent(inference, "int8_model_path", &config->inference.int8_model_path);
    ReadIfPresent(inference, "engine_path", &config->inference.engine_path);
    ReadBoolIfPresent(
        inference,
        "trt_require_prebuilt_engine",
        &config->inference.trt_require_prebuilt_engine);
    ReadIfPresent(inference, "openvino_model_path", &config->inference.openvino_model_path);
    ReadIfPresent(inference, "ncnn_param_path", &config->inference.ncnn_param_path);
    ReadIfPresent(inference, "ncnn_bin_path", &config->inference.ncnn_bin_path);
    ReadIfPresent(inference, "num_threads", &config->inference.num_threads);
    ReadIfPresent(inference, "positive_threshold", &config->inference.positive_threshold);
    ReadIfPresent(inference, "weak_threshold", &config->inference.weak_threshold);
    ReadIfPresent(inference, "max_candidates", &config->inference.max_candidates);
  }

  const cv::FileNode debug = fs["debug"];
  if (!debug.empty()) {
    ReadUInt32IfPresent(debug, "modules_mask", &config->debug.modules_mask);
  }

  return true;
}

}  // namespace pellet
