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
    ReadBoolIfPresent(motion, "adaptive_threshold", &config->motion.adaptive_threshold);
    ReadIfPresent(motion, "diff_threshold", &config->motion.diff_threshold);
    ReadIfPresent(motion, "diff_threshold_min", &config->motion.diff_threshold_min);
    ReadIfPresent(motion, "diff_threshold_max", &config->motion.diff_threshold_max);
    ReadBoolIfPresent(motion, "morph_enable", &config->motion.morph_enable);
    ReadIfPresent(motion, "morph_type", &config->motion.morph_type);
    ReadIfPresent(motion, "morph_kernel", &config->motion.morph_kernel);
    ReadIfPresent(motion, "morph_iters", &config->motion.morph_iters);
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
    ReadIfPresent(inference, "trt_engine_path", &config->inference.trt_engine_path);
    ReadIfPresent(inference, "openvino_model_path", &config->inference.openvino_model_path);
    ReadIfPresent(inference, "ncnn_param_path", &config->inference.ncnn_param_path);
    ReadIfPresent(inference, "ncnn_bin_path", &config->inference.ncnn_bin_path);
    ReadIfPresent(inference, "num_threads", &config->inference.num_threads);
    ReadBoolIfPresent(inference, "use_fp16", &config->inference.use_fp16);
    ReadIfPresent(inference, "positive_threshold", &config->inference.positive_threshold);
    ReadIfPresent(inference, "weak_threshold", &config->inference.weak_threshold);
    ReadIfPresent(inference, "max_candidates", &config->inference.max_candidates);
  }

  const cv::FileNode debug = fs["debug"];
  if (!debug.empty()) {
    ReadBoolIfPresent(debug, "enable", &config->debug.enable);
    ReadIfPresent(debug, "level", &config->debug.level);
    ReadUInt32IfPresent(debug, "modules_mask", &config->debug.modules_mask);
  }

  return true;
}

}  // namespace pellet
