#include "pellet/config.h"

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
    ReadIfPresent(camera, "source", &config->camera.source);
    ReadIfPresent(camera, "width", &config->camera.width);
    ReadIfPresent(camera, "height", &config->camera.height);
    ReadIfPresent(camera, "fps", &config->camera.fps);
    ReadBoolIfPresent(camera, "set_manual_exposure", &config->camera.set_manual_exposure);
    ReadIfPresent(camera, "exposure", &config->camera.exposure);
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
    ReadBoolIfPresent(debug, "show_window", &config->debug.show_window);
    ReadBoolIfPresent(debug, "show_mask", &config->debug.show_mask);
    ReadBoolIfPresent(debug, "show_morphology", &config->debug.show_morphology);
  }

  return true;
}

}  // namespace pellet
