# rm_aim_for_sm_pellets

面向 Detector 的 RoboMaster 小弹丸检测项目。

## 构建

```bash
cmake -S . -B build
cmake --build build -j
```

## 运行

```bash
./build/pellet_detector config/pellet.yaml
```

## Recorder 使用说明

`recorder` 用于把相机原始帧按二进制格式持续落盘，便于离线复现与数据回放。

常用命令：

```bash
# 默认读取 config/camera.yaml，输出到 logs/recorder/frames.rec
./build/recorder

# 指定相机配置和输出文件
./build/recorder --camera-config config/camera.yaml --output logs/recorder/frames.rec

# 限时录制 30 秒
./build/recorder --duration-sec 30

# 最多录制 5000 帧
./build/recorder --max-frames 5000

# 打开逐帧日志
./build/recorder --verbose
```

参数说明：

- `--camera-config <path>`：相机配置文件路径（默认 `config/camera.yaml`）。
- `--output <path>`：录制输出路径（默认 `logs/recorder/frames.rec`）。
- `--duration-sec <sec>`：最大录制时长，`0` 表示不限时。
- `--max-frames <n>`：最大录制帧数，`0` 表示不限帧。
- `--verbose`：输出逐帧日志。
- `--help`：打印帮助。

运行中会每 1 秒输出统计（fps、吞吐、累计帧数、丢帧数）；`Ctrl+C` 可优雅停止并自动收尾。

## 训练脚本功能（train/train_tiny_CNN.py）

训练脚本支持 5 种模式：`prepare`、`train`、`export`、`infer`、`all`。

- `prepare`：仅用 detector pipeline 处理原始数据，生成训练/验证样本。
- `train`：训练 TinyCNN（会先按配置准备数据）。
- `export`：导出 ONNX 到 `model/pellet_cls.onnx`。
- `infer`：对单张图做推理验证。
- `all`：执行训练 + 导出。

### 数据来源与对齐

当 `train/config/train.yaml` 里 `dataset.use_detector_pipeline: true` 时，训练输入会先经过与推理一致的 detector pipeline 处理：

- 灰度 + Gaussian
- 三帧差
- 二值化
- 开运算
- 连通域候选筛选
- ROI 裁剪并缩放到 `model.input_size`（默认 32）

这样训练输入与线上 detector 的 ROI 输入分布保持一致。

### 数据目录

原始数据目录（输入）：

- `train/dataset_raw/train/background`
- `train/dataset_raw/train/projectile`
- `train/dataset_raw/val/background`
- `train/dataset_raw/val/projectile`

处理后目录（训练实际使用）：

- `train/dataset/train/background`
- `train/dataset/train/projectile`
- `train/dataset/val/background`
- `train/dataset/val/projectile`

### 常用命令

```bash
# 1) 仅准备数据（跑 detector pipeline）
python3 train/train_tiny_CNN.py --config train/config/train.yaml --mode prepare

# 2) 训练（默认会准备数据）
python3 train/train_tiny_CNN.py --config train/config/train.yaml --mode train

# 3) 导出 ONNX
python3 train/train_tiny_CNN.py --config train/config/train.yaml --mode export

# 4) 单图推理验证
python3 train/train_tiny_CNN.py --config train/config/train.yaml --mode infer --image path/to/img.png
```

也可用封装脚本：

```bash
./scripts/run_train.sh
python3 tools/export_model.py
```

## 推理后端

- `mock`
- `onnx` / `onnxruntime`
- `tensorrt` / `trt`
- `openvino` / `ov`
- `ncnn`

除 `mock` 外，其它后端都依赖对应运行库和模型文件；若缺失，detector 初始化会失败。

## 目录结构

- `include/pellet`：按模块划分的对外头文件。
- `src`：detector、图像处理、推理和工具模块实现。
- `tests`：单元测试与集成测试。
- `tools`：基准测试与模型导出等工具程序/脚本。
- `scripts`：运行与性能分析辅助脚本。

## 检测流程（当前实现）

Camera  
→ 灰度 + Gaussian  
→ 三帧差（`D = (D1 & D2) | (D1 > T_high)`）  
→ 二值化  
→ 形态学（`open/close` 可配置）  
→ 连通域提取  
→ 候选过滤（面积、长宽比、`extent`、`local_contrast`、`motion_score`）  
→ **Pre-NMS 限流**（`motion.max_candidates`，工程上限制为 `<=20`）  
→ 候选去重 NMS（可开关，IoU 可配置）  
→ **Final Top-K**（`inference.max_candidates`，建议 `8~12`）  
→ ROI 裁剪并缩放到 `32x32`  
→ Tiny CNN 二分类  
→ 输出 `(frame_id, timestamp_ms, center, bbox, score)`

## 关键参数说明（config/pellet.yaml）

- `motion.max_candidates`：Pre-NMS 候选上限（`<=20`）。
- `inference.max_candidates`：Final Top-K 上限（NMS 后再截断，建议 `8~12`）。
- `motion.contrast_min`：候选局部对比度下限，抑制反光类误检。
- `motion.motion_score_min`：候选运动强度下限，抑制静态噪点。
- `motion.nms_enable`：是否启用 NMS（`0/1`）。
- `motion.nms_iou`：NMS IoU 阈值（常用 `0.2~0.3`，默认 `0.25`）。
- `debug.enable`：统一 debug 开关（`0/1`）。
- `debug.level`：分级开关（`0`=采集日志，`1`=采集+pipeline统计+线程状态，`>=2`=全部调试特性）。
- `debug.modules_mask`：手动位掩码覆盖（非 0 时优先于 `level`）。

说明：`motion.max_candidates` 与 `inference.max_candidates` 语义分离，前者用于 NMS 之前控制计算量，后者用于 NMS 之后控制最终送入 ROI/CNN 的数量。

## Debug 配置

- 仅支持统一 debug 配置：`debug.enable`、`debug.level`、`debug.modules_mask`。
- `modules_mask` 非 0 时优先级最高；否则由 `enable + level` 决定启用项。
