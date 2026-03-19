#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create masks_vis/ overlays from a BundleSDF custom dataset."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Dataset directory containing rgb/ and masks/.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write visualizations to. Defaults to <data_dir>/masks_vis.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay opacity in [0, 1].",
    )
    parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        default=(255, 0, 0),
        metavar=("B", "G", "R"),
        help="Overlay color in BGR order. Default is blue: 255 0 0.",
    )
    return parser.parse_args()


def load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if mask.ndim == 3:
        mask = (mask.sum(axis=-1) > 0).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255
    return mask


def build_overlay(color: np.ndarray, mask: np.ndarray, overlay_bgr: np.ndarray, alpha: float) -> np.ndarray:
    if mask.shape[:2] != color.shape[:2]:
        mask = cv2.resize(mask, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask > 0
    vis = color.copy()
    if not np.any(mask_bool):
        return vis

    tinted = vis.astype(np.float32)
    tinted[mask_bool] = (1.0 - alpha) * tinted[mask_bool] + alpha * overlay_bgr
    vis = np.clip(tinted, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
    return vis


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    rgb_dir = data_dir / "rgb"
    masks_dir = data_dir / "masks"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else data_dir / "masks_vis"
    )

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")

    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    overlay_bgr = np.array(args.color, dtype=np.float32)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not rgb_files:
        raise RuntimeError(f"No RGB images found in: {rgb_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for rgb_path in rgb_files:
        mask_path = masks_dir / rgb_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {rgb_path.name}: {mask_path}")

        color = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if color is None:
            raise FileNotFoundError(f"Failed to read RGB image: {rgb_path}")

        mask = load_mask(mask_path)
        vis = build_overlay(color, mask, overlay_bgr, alpha)

        out_path = output_dir / rgb_path.name
        if not cv2.imwrite(str(out_path), vis):
            raise RuntimeError(f"Failed to write visualization: {out_path}")
        written += 1

    print(f"Wrote {written} mask visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
