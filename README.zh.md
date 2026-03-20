# BundleSDF 中文使用说明

[English README](./readme.md)

## 简介
BundleSDF 用于从单目 RGB-D 视频中同时完成未知刚体的 6D 位姿跟踪和三维重建。

- 论文: [BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects](https://arxiv.org/abs/2303.14158)
- 项目主页: <https://bundlesdf.github.io/>

## 本地快速开始
如果你已经在本机准备好了 conda 环境，可以直接使用。当前机器验证通过的环境名是 `bundlesdf`。

```bash
conda activate bundlesdf
cd /path/to/BundleSDF
```

如果要处理 Intel RealSense 的 `.bag` 录像，还需要在同一个环境中安装：

```bash
pip install pyrealsense2
```

运行 `run_custom.py` 前，还需要准备 LoFTR 权重。至少保证下面这个文件存在：

```text
BundleTrack/LoFTR/weights/outdoor_ds.ckpt
```

如果你想用仓库里已经接好的 XMem 包装器做在线分割，还需要把 XMem 代码和权重放到：

```text
BundleTrack/XMem/
  └── saves/XMem.pth
```

## 自定义数据格式
`run_custom.py` 读取的是已经展开好的数据目录，而不是 `.bag` 文件。目录格式如下：

```text
root
  ├── rgb/      # PNG 彩色图
  ├── depth/    # PNG 深度图，单位 mm，uint16
  ├── masks/    # PNG 掩码，0 为背景，非 0 为前景
  └── cam_K.txt # 3x3 相机内参
```

## RealSense `.bag` 转换方法
仓库里已经提供了 `convert_realsense_bag.py`，会把 `.bag` 转成对齐后的 RGB-D 数据集：

```bash
python convert_realsense_bag.py \
  --bag /path/to/sequence.bag \
  --output_dir /path/to/sequence
```

转换后的输出目录会包含：

```text
/path/to/sequence
  ├── rgb/
  ├── depth/
  ├── masks/
  └── cam_K.txt
```

说明：
- `depth/` 中保存的是已经对齐到彩色图的 `uint16` 深度图，单位为毫米。
- 默认生成的 `masks/` 是“有效深度区域”的占位掩码，方便先把目录结构补齐并先跑通流程。
- 这个占位掩码不是“目标物体分割结果”。如果直接拿来做单物体跟踪，桌面、手和其他有深度的区域也会被当成前景。
- 如果想获得稳定的单物体跟踪与重建结果，请使用真实物体掩码，或者至少准备首帧目标 mask 再交给 XMem 向后传播。
- `--output_dir` 必须是空目录，或者目录原本不存在。

如果你只想导出 `rgb/`、`depth/` 和 `cam_K.txt`，不生成占位 `masks/`，可以加上：

```bash
python convert_realsense_bag.py \
  --bag /path/to/sequence.bag \
  --output_dir /path/to/sequence \
  --no_masks
```

## 掩码准备与可视化
这个工作区里现在支持 3 种常见用法：

### 1. 已经有逐帧物体掩码
目录中提前准备好完整的 `masks/*.png`，运行时使用 `--use_segmenter 0`。

### 2. 只有首帧物体掩码
如果只准备了 `masks/000000.png`，可以打开 XMem 包装器，让它自动向后传播分割结果：

- 启动命令里使用 `--use_segmenter 1`
- 首帧 `masks/000000.png` 必须是真实目标物体 mask
- 传播结果会写到输入数据目录旁边的 `masks_xmem/`

### 3. 只有占位掩码
`convert_realsense_bag.py` 默认生成的占位 `masks/` 只适合做流程联调，不适合真正的单物体结果。

如果你想把当前 `masks/` 叠加显示成 `masks_vis/` 方便检查，可以使用：

```bash
python create_masks_vis.py \
  --data_dir /path/to/sequence
```

补充说明：
- `masks_vis/` 只是可视化结果，不会自动改进分割质量。
- `create_masks_vis.py` 默认生成蓝色半透明叠加图，也可以通过 `--alpha` 和 `--color` 调整样式。

如果你想“点几下目标”自动生成首帧物体 mask，可以使用交互脚本：

```bash
python click_first_frame_mask.py \
  --data_dir /path/to/sequence \
  --frame 0
```

交互方式：
- 左键点击目标物体，添加前景点
- 右键点击背景区域，添加背景点
- 按 `Enter` 或 `r` 自动细化 mask
- 按 `s` 保存到 `masks/000000.png`

这个脚本会优先使用同名深度图做一个轻量深度先验，所以通常比纯 RGB 点选更稳一些。

## 运行方法
### 1. 跟踪与在线重建
如果你已经在 `masks/` 中准备好了逐帧物体掩码，运行时请使用 `--use_segmenter 0`：

```bash
python run_custom.py \
  --mode run_video \
  --video_dir /path/to/sequence \
  --out_folder /path/to/output \
  --use_segmenter 0 \
  --use_gui 0 \
  --debug_level 2
```

如果只准备了首帧物体掩码（例如 `masks/000000.png`），可以使用 XMem 包装器自动向后传播分割结果：

```bash
python run_custom.py \
  --mode run_video \
  --video_dir /path/to/sequence \
  --out_folder /path/to/output \
  --use_segmenter 1 \
  --use_gui 0 \
  --debug_level 2
```

如果显存比较紧张，可以在运行前加上：

```bash
export BUNDLESDF_LOFTR_BATCH_SIZE=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 2. 全局优化
```bash
python run_custom.py \
  --mode global_refine \
  --video_dir /path/to/sequence \
  --out_folder /path/to/output
```

默认会使用更省内存的 `memory` profile。如果你想切回更重、更慢但更接近原始仓库设置的版本，可以显式指定：

```bash
python run_custom.py \
  --mode global_refine \
  --video_dir /path/to/sequence \
  --out_folder /path/to/output \
  --global_refine_profile full
```

### 3. 可选：绘制位姿框
```bash
python run_custom.py \
  --mode draw_pose \
  --out_folder /path/to/output
```

## 当前机器上的示例
下面这组命令已经在当前机器上验证通过：

### RealSense `.bag` 转换与运行
```bash
conda activate bundlesdf
cd /home/ustczxh/humanoid/BundleSDF

python convert_realsense_bag.py \
  --bag /home/ustczxh/realsense/20260319_184534.bag \
  --output_dir /home/ustczxh/realsense/20260319_184534

python create_masks_vis.py \
  --data_dir /home/ustczxh/realsense/20260319_184534

python click_first_frame_mask.py \
  --data_dir /home/ustczxh/realsense/20260319_184534 \
  --frame 0

python run_custom.py \
  --mode run_video \
  --video_dir /home/ustczxh/realsense/20260319_184534 \
  --out_folder /home/ustczxh/realsense/output/20260319_184534 \
  --use_segmenter 1 \
  --use_gui 0 \
  --debug_level 2

python run_custom.py \
  --mode global_refine \
  --video_dir /home/ustczxh/realsense/20260319_184534 \
  --out_folder /home/ustczxh/realsense/output/20260319_184534
```

说明：
- `run_video` 现在默认不会再自动触发 `global_refine`，这样更符合“先跟踪、再按需精修”的预期，也能避免尾声阶段突然吃满内存。
- 如果你确实想在 `run_video` 结束后自动接着跑，可以加 `--auto_global_refine 1`。

如果你给 `masks/000000.png` 换成了真实物体首帧 mask，也可以打开 XMem：

```bash
export BUNDLESDF_LOFTR_BATCH_SIZE=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_custom.py \
  --mode run_video \
  --video_dir /home/ustczxh/realsense/20260319_184534 \
  --out_folder /home/ustczxh/realsense/output_keyboard_xmem \
  --use_segmenter 1 \
  --use_gui 0
```

### 示例数据 `milk`
```bash
conda activate bundlesdf
cd /home/ustczxh/humanoid/BundleSDF

python run_custom.py \
  --mode run_video \
  --video_dir /home/ustczxh/humanoid/BundleSDF/example_data/2022-11-18-15-10-24_milk \
  --out_folder /home/ustczxh/humanoid/BundleSDF/example_data/milk_output \
  --use_gui 0

python run_custom.py \
  --mode global_refine \
  --video_dir /home/ustczxh/humanoid/BundleSDF/example_data/2022-11-18-15-10-24_milk \
  --out_folder /home/ustczxh/humanoid/BundleSDF/example_data/milk_output
```

## 输出结果
运行结束后，结果会写入 `out_folder`，常见内容包括：

- `ob_in_cam/`: 每帧物体位姿
- `mesh_cleaned.obj`: 清理后的几何网格
- `textured_mesh.obj`: 带纹理的重建网格
- `masks_xmem/`: 如果开启了 XMem，会把传播得到的分割结果写到输入数据目录旁边，方便检查

## 注意事项
- `run_custom.py` 不能直接把 `.bag` 路径传给 `--video_dir`。
- `create_masks_vis.py` 只负责把现有 `masks/` 画出来，不会自动帮你识别目标物体。
- `click_first_frame_mask.py` 只负责生成首帧目标 mask；如果后续要自动传播，需要再配合 `--use_segmenter 1` 和 XMem。
- 如果没有额外接入 XMem 等在线分割模块，就保持 `--use_segmenter 0`，并提前准备好逐帧 `masks/`。
- 如果开启 `--use_segmenter 1`，请至少保证 `masks/000000.png` 是真实目标物体 mask，而不是占位的有效深度图。
- `global_refine` 默认使用更省内存的 `memory` profile；如果机器内存和显存都比较充足，再考虑切到 `--global_refine_profile full`。
- 默认配置假设目标相关深度范围比较近；如果场景更远，可以调整 `BundleTrack/config_ho3d.yml` 中的深度参数。
