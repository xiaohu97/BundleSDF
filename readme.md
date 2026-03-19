# BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects

[中文说明](./README.zh.md)

This is an implementation of our paper published in CVPR 2023

[[Arxiv](https://arxiv.org/abs/2303.14158)] [[Project page](https://bundlesdf.github.io/)] [[Supplemental video](https://www.youtube.com/watch?v=5PymzKbKv8w/)]

# Abstract
We present a near real-time method for 6-DoF tracking of an unknown object from a monocular RGBD video sequence, while simultaneously performing neural 3D reconstruction of the object. Our method works for arbitrary rigid objects, even when visual texture is largely absent. The object is assumed to be segmented in the first frame only. No additional information is required, and no assumption is made about the interaction agent. Key to our method is a Neural Object Field that is learned concurrently with a pose graph optimization process in order to robustly accumulate information into a consistent 3D representation capturing both geometry and appearance. A dynamic pool of posed memory frames is automatically maintained to facilitate communication between these threads. Our approach handles challenging sequences with large pose changes, partial and full occlusion, untextured surfaces, and specular highlights. We show results on HO3D, YCBInEOAT, and BEHAVE datasets, demonstrating that our method significantly outperforms existing approaches.

<img src="./media/problem_setup_c.gif" width="80%">

<img src="./media/preview_results_c.gif" width="80%">

<img src="./media/driller.gif" width="80%">

# Bibtex
```bibtex
@InProceedings{bundlesdfwen2023,
author        = {Bowen Wen and Jonathan Tremblay and Valts Blukis and Stephen Tyree and Thomas M\"{u}ller and Alex Evans and Dieter Fox and Jan Kautz and Stan Birchfield},
title         = {{BundleSDF}: {N}eural 6-{DoF} Tracking and {3D} Reconstruction of Unknown Objects},
booktitle     = {CVPR},
year          = {2023},
}
```

# Data download
- Download pretrained [weights of segmentation network](https://drive.google.com/file/d/1MEZvjbBdNAOF7pXcq6XPQduHeXB50VTc/view?usp=share_link), and put it under
`./BundleTrack/XMem/saves/XMem.pth`

- Download pretrained [weights of LoFTR outdoor_ds.ckpt](https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp), and put it under
`./BundleTrack/LoFTR/weights/outdoor_ds.ckpt`

- Download HO3D data. We provide the augmented data that you can download [here](https://drive.google.com/drive/folders/1Wk-HZDvUExyUrRn7us4WWEbHnnFHgOAX?usp=share_link). Then download YCB-Video object models from [here](https://drive.google.com/file/d/1-1m7qMMyUHYLhaRiQBbsSRMt5dMRX4jD/view?usp=share_link). Finally, make sure the structure is like below, and update your root path of `HO3D_ROOT` at the top of `BundleTrack/scripts/data_reader.py`
  ```
  HO3D_v3
    ├── evaluation
    ├── models
    └── masks_XMem
  ```


# Docker/Environment setup
- Build the docker image (this only needs to do once and can take some time).
```
cd docker
docker build --network host -t nvcr.io/nvidian/bundlesdf .
```

- Start a docker container the first time
```
cd docker && bash run_container.sh

# Inside docker container, compile the packages which are machine dependent
bash build.sh
```

# Quick start without Docker
The commands below are the shortest local workflow for a conda environment. On this machine, the tested environment name is `bundlesdf`.

```bash
conda activate bundlesdf
cd /path/to/BundleSDF
```

If you want to convert Intel RealSense `.bag` recordings, make sure `pyrealsense2` is available in the same environment:

```bash
pip install pyrealsense2
```

LoFTR weights are required before running `run_custom.py`. The download folder should contain at least `outdoor_ds.ckpt` under:

```text
BundleTrack/LoFTR/weights/
```

# RealSense .bag workflow
`run_custom.py` expects an extracted dataset folder, not a `.bag` file. For RealSense recordings, first convert the bag into aligned RGB-D frames:

```bash
python convert_realsense_bag.py \
  --bag /path/to/sequence.bag \
  --output_dir /path/to/sequence
```

This writes:

```text
/path/to/sequence
  ├── rgb/
  ├── depth/
  ├── masks/
  └── cam_K.txt
```

Notes:
- `depth/` is saved as aligned `uint16` depth in millimeters.
- `masks/` are placeholder valid-depth masks by default. They are only meant to bootstrap the folder structure or do a quick pipeline smoke test.
- Those placeholder masks are not object segmentation. If you use them directly for single-object tracking, other valid-depth regions such as the table, hand, or nearby clutter can also be treated as foreground.
- For real object tracking/reconstruction, replace `masks/` with actual object masks, or at least provide a true first-frame object mask and let XMem propagate it.
- The output directory must be empty or not exist before conversion.

If you only want `rgb/`, `depth/`, and `cam_K.txt`, skip placeholder masks entirely:

```bash
python convert_realsense_bag.py \
  --bag /path/to/sequence.bag \
  --output_dir /path/to/sequence \
  --no_masks
```

# Run on your custom data
- Prepare your RGBD video folder as below (also refer to the example milk data). You can find an [example milk data here](https://drive.google.com/file/d/1akutk_Vay5zJRMr3hVzZ7s69GT4gxuWN/view?usp=share_link) for testing.
```
root
  ├──rgb/    (PNG files)
  ├──depth/  (PNG files, stored in mm, uint16 format. Filename same as rgb)
  ├──masks/       (PNG files. Filename same as rgb. 0 is background. Else is foreground)
  └──cam_K.txt   (3x3 intrinsic matrix, use space and enter to delimit)
```

## Masks, XMem, and visualization
This workspace currently supports three common mask workflows:

1. Precomputed masks for every frame
Keep full `masks/*.png` in the dataset and run with `--use_segmenter 0`.

2. Only a first-frame object mask
Provide a real object mask in `masks/000000.png`, then run with `--use_segmenter 1` so XMem can propagate the segmentation through the sequence.

3. Placeholder valid-depth masks from `convert_realsense_bag.py`
These are useful for pipeline bring-up only. They are not a substitute for object segmentation.

The original BundleSDF release does not vendor [XMem](https://github.com/hkchengrex/XMem) directly. In this workspace, the expected layout for the optional online segmenter is:
```text
BundleTrack/XMem/
  └── saves/XMem.pth
```
With that in place, `segmentation_utils.py` can use XMem to propagate a first-frame object mask through the sequence. The generated predictions are also written to `masks_xmem/` beside your input data for inspection.

If you want a quick overlay view of the masks you already have, this workspace also includes:

```bash
python create_masks_vis.py \
  --data_dir /path/to/sequence
```

This creates `masks_vis/` by drawing the current masks over `rgb/`. It is a visualization helper only, and does not improve the masks themselves.

- Run your RGBD video (specify the video_dir and your desired output path). There are 3 steps. Note we assume the max relevant depth in the demo data <1. If this is not the case for you, change it [here](https://github.com/NVlabs/BundleSDF/blob/master/BundleTrack/config_ho3d.yml#L16)
```
# 1) Run joint tracking and reconstruction.
# For precomputed per-frame masks in masks/, use --use_segmenter 0.
python run_custom.py --mode run_video --video_dir /path/to/sequence --out_folder /path/to/output --use_segmenter 0 --use_gui 0 --debug_level 2

# Or, if only the first-frame object mask is prepared in masks/000000.png,
# let XMem propagate the mask to later frames. The first-frame mask must be
# a real object mask, not the placeholder valid-depth mask from bag conversion.
python run_custom.py --mode run_video --video_dir /path/to/sequence --out_folder /path/to/output --use_segmenter 1 --use_gui 0 --debug_level 2

# 2) Run global refinement post-processing to refine the mesh
python run_custom.py --mode global_refine --video_dir /path/to/sequence --out_folder /path/to/output

# 3) (Optional) If you want to draw the oriented bounding box to visualize the pose, similar to our demo
python run_custom.py --mode draw_pose --out_folder /path/to/output
```

If you are tight on GPU memory, the following environment variables can help on smaller cards:

```bash
export BUNDLESDF_LOFTR_BATCH_SIZE=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- Example on this machine:
```
python convert_realsense_bag.py --bag /home/ustczxh/realsense/20260319_184534.bag --output_dir /home/ustczxh/realsense/20260319_184534
python create_masks_vis.py --data_dir /home/ustczxh/realsense/20260319_184534
python run_custom.py --mode run_video --video_dir /home/ustczxh/realsense/20260319_184534 --out_folder /home/ustczxh/realsense/output --use_gui 0
python run_custom.py --mode global_refine --video_dir /home/ustczxh/realsense/20260319_184534 --out_folder /home/ustczxh/realsense/output
```

- Example with a first-frame object mask plus XMem propagation on this machine:
```
export BUNDLESDF_LOFTR_BATCH_SIZE=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_custom.py --mode run_video --video_dir /home/ustczxh/realsense/20260319_184534 --out_folder /home/ustczxh/realsense/output_keyboard_xmem --use_segmenter 1 --use_gui 0
```

- Finally the results will be dumped in the `out_folder`. Common outputs include:
- `ob_in_cam/` for tracked poses
- `mesh_cleaned.obj` for the cleaned geometry
- `textured_mesh.obj` for the textured reconstruction mesh
- `masks_xmem/` beside the input data if XMem propagation is enabled

<img src="./media/milk_jug.gif" height="400">


# Run on HO3D dataset
```
# Run BundleSDF to get the pose and reconstruction results
python run_ho3d.py --video_dirs /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/SM1 --out_dir /home/bowen/debug/ho3d_ours

# Benchmark the output results
python benchmark_ho3d.py --video_dirs /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/SM1 --out_dir /home/bowen/debug/ho3d_ours
```


# Acknowledgement

We would like to thank Jeff Smith for helping with the code release. Marco Foco and his team for providing the test data on the static scene.


# Contact
For questions, please contact Bowen Wen (bowenw@nvidia.com)
