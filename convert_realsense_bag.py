#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyrealsense2 is required to convert .bag files.\n"
        "Please run this script inside the `bundlesdf` conda environment."
    ) from exc


def ensure_empty_dir(path: Path):
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Output directory already exists and is not empty: {path}\n"
            "Please choose a new directory or remove the old contents first."
        )
    path.mkdir(parents=True, exist_ok=True)


def write_camera_matrix(output_dir: Path, intrinsics):
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.savetxt(output_dir / "cam_K.txt", camera_matrix, fmt="%.8f")


def convert_color_to_bgr(color_image: np.ndarray, color_format):
    if color_format == rs.format.rgb8:
        return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    if color_format == rs.format.rgba8:
        return cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGR)
    if color_format == rs.format.bgr8:
        return color_image
    if color_format == rs.format.bgra8:
        return cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    if color_format == rs.format.y8:
        return cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported color format in bag file: {color_format}")


def save_png(path: Path, image: np.ndarray):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def convert_bag(
    bag_path: Path,
    output_dir: Path,
    start_time: float | None,
    end_time: float | None,
    create_masks: bool,
):
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    masks_dir = output_dir / "masks"
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    if create_masks:
        masks_dir.mkdir(exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    frame_idx = 0
    first_timestamp_ms = None
    intrinsics_written = False

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError as exc:
                if frame_idx > 0 and (
                    playback.current_status() == rs.playback_status.stopped
                    or "Frame didn't arrive within" in str(exc)
                ):
                    break
                raise RuntimeError(f"Failed while reading {bag_path}: {exc}") from exc

            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            if first_timestamp_ms is None:
                first_timestamp_ms = aligned_frames.get_timestamp()
            rel_time_s = (aligned_frames.get_timestamp() - first_timestamp_ms) / 1000.0

            if start_time is not None and rel_time_s < start_time:
                continue
            if end_time is not None and rel_time_s > end_time:
                break

            color_image = np.asanyarray(color_frame.get_data())
            color_image = convert_color_to_bgr(color_image, color_frame.profile.format())

            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_mm = np.rint(depth_raw.astype(np.float32) * depth_scale * 1000.0).astype(
                np.uint16
            )

            if not intrinsics_written:
                intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                write_camera_matrix(output_dir, intrinsics)
                intrinsics_written = True

            stem = f"{frame_idx:06d}"
            save_png(rgb_dir / f"{stem}.png", color_image)
            save_png(depth_dir / f"{stem}.png", depth_mm)

            if create_masks:
                mask = np.where(depth_mm > 0, 255, 0).astype(np.uint8)
                save_png(masks_dir / f"{stem}.png", mask)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Converted {frame_idx} aligned RGB-D frames...")
    finally:
        pipeline.stop()

    if frame_idx == 0:
        raise RuntimeError("No aligned RGB-D frames were written. Check the selected time range.")

    return frame_idx


def main():
    parser = argparse.ArgumentParser(
        description="Convert an Intel RealSense .bag file into BundleSDF custom-data layout."
    )
    parser.add_argument("--bag", required=True, help="Path to the RealSense .bag file")
    parser.add_argument(
        "--output_dir",
        help="Directory to write the converted dataset to. Defaults to <bag-without-extension>/",
    )
    parser.add_argument("--start_time", type=float, default=None, help="Optional start time in seconds")
    parser.add_argument("--end_time", type=float, default=None, help="Optional end time in seconds")
    parser.add_argument(
        "--no_masks",
        action="store_true",
        help="Skip creating placeholder valid-depth masks",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else bag_path.with_suffix("")
    )
    ensure_empty_dir(output_dir)

    num_frames = convert_bag(
        bag_path=bag_path,
        output_dir=output_dir,
        start_time=args.start_time,
        end_time=args.end_time,
        create_masks=not args.no_masks,
    )

    print(f"Converted {num_frames} aligned RGB-D frame pairs to: {output_dir}")
    if args.no_masks:
        print("No masks were generated. Provide masks before running run_custom.py with --use_segmenter 0.")
    else:
        print(
            "Generated placeholder valid-depth masks in masks/. "
            "Replace them with object masks for better results."
        )


if __name__ == "__main__":
    main()
