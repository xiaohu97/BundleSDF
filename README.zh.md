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
- 默认生成的 `masks/` 是“有效深度区域”的占位掩码，方便先把流程跑通。
- 如果想获得更稳定的单物体跟踪与重建结果，建议把 `masks/` 替换成真实的目标分割掩码。
- `--output_dir` 必须是空目录，或者目录原本不存在。

## 运行方法
### 1. 跟踪与在线重建
如果你已经在 `masks/` 中准备好了掩码，运行时请使用 `--use_segmenter 0`：

```bash
python run_custom.py \
  --mode run_video \
  --video_dir /path/to/sequence \
  --out_folder /path/to/output \
  --use_segmenter 0 \
  --use_gui 0 \
  --debug_level 2
```

### 2. 全局优化
```bash
python run_custom.py \
  --mode global_refine \
  --video_dir /path/to/sequence \
  --out_folder /path/to/output
```

### 3. 可选：绘制位姿框
```bash
python run_custom.py \
  --mode draw_pose \
  --out_folder /path/to/output
```

## 当前机器上的示例
下面这组命令已经在当前机器上验证通过：

```bash
conda activate bundlesdf
cd /home/ustczxh/humanoid/BundleSDF

python convert_realsense_bag.py \
  --bag /home/ustczxh/realsense/20260319_184534.bag \
  --output_dir /home/ustczxh/realsense/20260319_184534

python run_custom.py \
  --mode run_video \
  --video_dir /home/ustczxh/realsense/20260319_184534 \
  --out_folder /home/ustczxh/realsense/output \
  --use_gui 0

python run_custom.py \
  --mode global_refine \
  --video_dir /home/ustczxh/realsense/20260319_184534 \
  --out_folder /home/ustczxh/realsense/output
```

```
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

## 注意事项
- `run_custom.py` 不能直接把 `.bag` 路径传给 `--video_dir`。
- 如果没有额外接入 XMem 等在线分割模块，就保持 `--use_segmenter 0`，并提前准备好 `masks/`。
- 默认配置假设目标相关深度范围比较近；如果场景更远，可以调整 `BundleTrack/config_ho3d.yml` 中的深度参数。
