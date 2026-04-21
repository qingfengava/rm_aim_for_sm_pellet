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


Camera
→ 灰度 + 轻量去噪(3x3 Gaussian)
→ 三帧差（D = (D1 & D2) | (D1 > T_high)） + 自适应双阈值(T_low/T_high)
→ 二值掩码(滞回阈值，而不是单阈值)（类似canny）
→ 形态学(优先 close；open 仅在噪点多时启用)（现改为可选项，怀疑容易筛调真目标）
→ 连通域 提取+ 几何/光度过滤（候选筛选里加 长宽比/extent/局部对比度，比只看面积和圆形度更抗反光误检。）
→ 候选去重(NMS) （降低推理占用）+ Top-K(建议 8~12)（限制最大候选数（≤20））
→ ROI裁剪(32x32，按运动质心居中)（动态裁减，后输入网络统一缩放为32x32）
→ Tiny CNN(INT8量化，批量推理)
→ 置信度双阈值 + 2/3帧短时投票（减少抖动）
→ 输出 (frame_id, timestamp, center, bbox, score)
