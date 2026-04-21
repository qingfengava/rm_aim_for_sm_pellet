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

## 训练与导出

- 训练配置文件：`train/config/train.yaml`
- 训练集路径：`train/dataset/train/{background,projectile}`
- 验证集路径：`train/dataset/val/{background,projectile}`
- 权重输出：`model/pellet_cls.pth`
- ONNX 输出：`model/pellet_cls.onnx`

```bash
./scripts/run_train.sh
```

仅导出 ONNX：

```bash
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
→ 三帧差 + 自适应双阈值(T_low/T_high)
→ 二值掩码(滞回阈值，而不是单阈值)（类似canny）
→ 形态学(优先 close；open 仅在噪点多时启用)（现改为可选项，怀疑容易筛调真目标）
→ 连通域 提取+ 几何/光度过滤（候选筛选里加 长宽比/extent/局部对比度，比只看面积和圆形度更抗反光误检。）
→ 候选去重(NMS) （降低推理占用）+ Top-K(建议 8~12)（限制最大候选数（≤20））
→ ROI裁剪(32x32，按运动质心居中)（动态裁减，后输入网络统一缩放为32x32）
→ Tiny CNN(INT8量化，批量推理)
→ 置信度双阈值 + 2/3帧短时投票（减少抖动）
→ 输出 (frame_id, timestamp, center, bbox, score)
