#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "BundleSDF First-Frame Mask Tool"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a first-frame object mask with a few mouse clicks. "
            "Left click adds foreground points, right click adds background points."
        )
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="BundleSDF custom-data directory containing rgb/ and masks/.",
    )
    parser.add_argument(
        "--frame",
        default="0",
        help="Frame index in sorted rgb/*.png, or an exact file stem such as 000000.",
    )
    parser.add_argument(
        "--point_radius",
        type=int,
        default=10,
        help="Radius in pixels for foreground/background hints.",
    )
    parser.add_argument(
        "--depth_tolerance_mm",
        type=float,
        default=120.0,
        help="Depth prior tolerance around clicked foreground points in millimeters.",
    )
    parser.add_argument(
        "--max_display_size",
        type=int,
        default=1400,
        help="Resize the preview window so its longest edge is at most this many pixels.",
    )
    parser.add_argument(
        "--no_depth_prior",
        action="store_true",
        help="Disable the optional depth prior when a depth image is available.",
    )
    return parser.parse_args()


def resolve_rgb_path(rgb_dir: Path, frame_arg: str) -> Path:
    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in: {rgb_dir}")

    frame_arg = frame_arg.strip()
    if frame_arg.isdigit():
        index = int(frame_arg)
        if 0 <= index < len(rgb_files):
            return rgb_files[index]

    if not frame_arg.endswith(".png"):
        frame_arg = f"{frame_arg}.png"
    direct = rgb_dir / frame_arg
    if direct.is_file():
        return direct

    raise FileNotFoundError(
        f"Could not resolve frame '{frame_arg}'. "
        f"Expected an index in [0, {len(rgb_files) - 1}] or an existing png stem."
    )


def load_depth(depth_path: Path):
    if not depth_path.is_file():
        return None
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.ndim != 2:
        return None
    return depth


def ensure_masks_dir(data_dir: Path) -> Path:
    masks_dir = data_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    return masks_dir


def backup_existing_mask(mask_path: Path):
    if not mask_path.is_file():
        return None
    backup_dir = mask_path.parent.parent / "masks_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{mask_path.stem}_{timestamp}.png"
    existing = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if existing is None:
        return None
    cv2.imwrite(str(backup_path), existing)
    return backup_path


def alpha_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    vis = image_bgr.copy().astype(np.float32)
    mask_bool = mask > 0
    if np.any(mask_bool):
        color = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        vis[mask_bool] = (1.0 - alpha) * vis[mask_bool] + alpha * color
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
    else:
        vis = vis.astype(np.uint8)
    return vis


def build_click_bbox(points, image_shape):
    h, w = image_shape[:2]
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    span_x = max(80, x_max - x_min)
    span_y = max(80, y_max - y_min)
    pad_x = max(30, int(round(span_x * 0.4)))
    pad_y = max(30, int(round(span_y * 0.4)))
    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(w - 1, x_max + pad_x)
    y1 = min(h - 1, y_max + pad_y)
    return x0, y0, x1, y1


def build_fg_core_mask(fg_points, image_shape, point_radius):
    h, w = image_shape[:2]
    core = np.zeros((h, w), dtype=np.uint8)
    if len(fg_points) >= 3:
        pts = np.array(fg_points, dtype=np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(core, hull, 255)
    else:
        x0, y0, x1, y1 = build_click_bbox(fg_points, image_shape)
        core[y0 : y1 + 1, x0 : x1 + 1] = 255

    for x, y in fg_points:
        cv2.circle(core, (x, y), max(2, point_radius), 255, thickness=-1)
    return core


def build_candidate_masks(fg_points, image_shape, point_radius):
    core = build_fg_core_mask(fg_points, image_shape, point_radius)
    ys, xs = np.where(core > 0)
    if len(xs) == 0:
        raise RuntimeError("Failed to build a foreground candidate region from the clicks.")

    span_x = int(xs.max() - xs.min() + 1)
    span_y = int(ys.max() - ys.min() + 1)
    long_span = max(span_x, span_y)

    local_margin = max(point_radius * 2, int(round(long_span * 0.06)))
    local_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * local_margin + 1, 2 * local_margin + 1),
    )
    local_region = cv2.dilate(core, local_kernel, iterations=1)

    candidate_margin = max(point_radius * 3, int(round(long_span * 0.16)))
    candidate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * candidate_margin + 1, 2 * candidate_margin + 1),
    )
    candidate = cv2.dilate(core, candidate_kernel, iterations=1)
    return core, local_region, candidate


def keep_clicked_components(mask: np.ndarray, fg_points):
    if not np.any(mask):
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep_ids = set()
    for x, y in fg_points:
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
            label = int(labels[y, x])
            if label > 0:
                keep_ids.add(label)
    if not keep_ids:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep_ids.add(largest)
    kept = np.isin(labels, list(keep_ids)).astype(np.uint8) * 255
    return kept


def refine_mask(image_bgr, depth_mm, fg_points, bg_points, point_radius, depth_tolerance_mm, use_depth_prior):
    if not fg_points:
        raise RuntimeError("Add at least one foreground point before refining.")

    h, w = image_bgr.shape[:2]
    grabcut_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    core_region, local_region, candidate_region = build_candidate_masks(
        fg_points=fg_points,
        image_shape=image_bgr.shape,
        point_radius=point_radius,
    )

    grabcut_mask[candidate_region > 0] = cv2.GC_PR_BGD
    grabcut_mask[core_region > 0] = cv2.GC_PR_FGD

    if use_depth_prior and depth_mm is not None:
        fg_depths = []
        for x, y in fg_points:
            if 0 <= y < depth_mm.shape[0] and 0 <= x < depth_mm.shape[1]:
                value = float(depth_mm[y, x])
                if value > 0:
                    fg_depths.append(value)
        if fg_depths:
            center_depth = float(np.median(fg_depths))
            valid_depth = depth_mm > 0
            close_depth = np.abs(depth_mm.astype(np.float32) - center_depth) <= depth_tolerance_mm
            prior_fg = (local_region > 0) & valid_depth & close_depth
            if int(prior_fg.sum()) > 128:
                grabcut_mask[prior_fg] = cv2.GC_PR_FGD

    for x, y in fg_points:
        cv2.circle(grabcut_mask, (x, y), point_radius, cv2.GC_FGD, thickness=-1)
    for x, y in bg_points:
        cv2.circle(grabcut_mask, (x, y), point_radius, cv2.GC_BGD, thickness=-1)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, grabcut_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)

    refined = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    refined[candidate_region == 0] = 0
    refined = keep_clicked_components(refined, fg_points)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)

    for x, y in fg_points:
        cv2.circle(refined, (x, y), max(2, point_radius // 2), 255, thickness=-1)
    for x, y in bg_points:
        cv2.circle(refined, (x, y), point_radius, 0, thickness=-1)

    return refined


@dataclass
class ClickHint:
    kind: str
    x: int
    y: int


class FirstFrameMaskTool:
    def __init__(self, image_bgr, depth_mm, rgb_path, mask_path, point_radius, depth_tolerance_mm, max_display_size, use_depth_prior):
        self.image_bgr = image_bgr
        self.depth_mm = depth_mm
        self.rgb_path = rgb_path
        self.mask_path = mask_path
        self.preview_path = mask_path.parent.parent / f"{mask_path.stem}_mask_preview.png"
        self.point_radius = point_radius
        self.depth_tolerance_mm = depth_tolerance_mm
        self.use_depth_prior = use_depth_prior
        self.hints = []
        self.mask = None
        self.status = "Left click: foreground, right click: background, Enter/r: refine, s: save, u: undo, c: clear, q: quit"

        h, w = image_bgr.shape[:2]
        self.display_scale = min(1.0, float(max_display_size) / float(max(h, w)))
        self.display_size = (int(round(w * self.display_scale)), int(round(h * self.display_scale)))
        self.display_image = cv2.resize(image_bgr, self.display_size, interpolation=cv2.INTER_AREA) if self.display_scale < 1.0 else image_bgr.copy()

    def _to_image_xy(self, x, y):
        if self.display_scale < 1.0:
            x = int(round(x / self.display_scale))
            y = int(round(y / self.display_scale))
        x = int(np.clip(x, 0, self.image_bgr.shape[1] - 1))
        y = int(np.clip(y, 0, self.image_bgr.shape[0] - 1))
        return x, y

    def _to_display_xy(self, x, y):
        if self.display_scale < 1.0:
            x = int(round(x * self.display_scale))
            y = int(round(y * self.display_scale))
        return x, y

    def _render(self):
        base = self.image_bgr if self.mask is None else alpha_overlay(self.image_bgr, self.mask)
        if self.display_scale < 1.0:
            canvas = cv2.resize(base, self.display_size, interpolation=cv2.INTER_AREA)
        else:
            canvas = base.copy()

        for hint in self.hints:
            dx, dy = self._to_display_xy(hint.x, hint.y)
            color = (0, 255, 0) if hint.kind == "fg" else (0, 0, 255)
            cv2.circle(canvas, (dx, dy), max(4, int(round(self.point_radius * self.display_scale))), color, thickness=-1)
            cv2.circle(canvas, (dx, dy), max(5, int(round((self.point_radius + 2) * self.display_scale))), (255, 255, 255), thickness=1)

        cv2.putText(canvas, Path(self.rgb_path).name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, self.status, (20, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _mouse_callback(self, event, x, y, _flags, _param):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        ix, iy = self._to_image_xy(x, y)
        kind = "fg" if event == cv2.EVENT_LBUTTONDOWN else "bg"
        self.hints.append(ClickHint(kind=kind, x=ix, y=iy))
        self.status = f"Added {'foreground' if kind == 'fg' else 'background'} point at ({ix}, {iy})"

    def _refine(self):
        fg_points = [(h.x, h.y) for h in self.hints if h.kind == "fg"]
        bg_points = [(h.x, h.y) for h in self.hints if h.kind == "bg"]
        self.mask = refine_mask(
            image_bgr=self.image_bgr,
            depth_mm=self.depth_mm,
            fg_points=fg_points,
            bg_points=bg_points,
            point_radius=self.point_radius,
            depth_tolerance_mm=self.depth_tolerance_mm,
            use_depth_prior=self.use_depth_prior,
        )
        area = int((self.mask > 0).sum())
        self.status = f"Refined mask with {len(fg_points)} foreground and {len(bg_points)} background clicks. Area={area} px"

    def _save(self):
        if self.mask is None:
            self._refine()
        backup_path = backup_existing_mask(self.mask_path)
        cv2.imwrite(str(self.mask_path), self.mask)
        preview = alpha_overlay(self.image_bgr, self.mask)
        cv2.imwrite(str(self.preview_path), preview)
        self.status = f"Saved mask to {self.mask_path}"
        print(f"Saved mask: {self.mask_path}")
        print(f"Saved preview: {self.preview_path}")
        if backup_path is not None:
            print(f"Backed up previous mask to: {backup_path}")

    def run(self):
        print("Controls:")
        print("  Left click : add foreground point")
        print("  Right click: add background point")
        print("  Enter / r  : refine mask")
        print("  u          : undo last point")
        print("  c          : clear all points and current mask")
        print("  s          : save mask to masks/<frame>.png")
        print("  q / Esc    : quit")
        print(f"Current frame: {self.rgb_path}")
        print(f"Target mask : {self.mask_path}")
        if self.depth_mm is None:
            print("Depth prior : disabled (depth image not found)")
        elif self.use_depth_prior:
            print(f"Depth prior : enabled, tolerance={self.depth_tolerance_mm:.1f} mm")
        else:
            print("Depth prior : disabled by --no_depth_prior")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        while True:
            cv2.imshow(WINDOW_NAME, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, ord("r")):
                try:
                    self._refine()
                except Exception as exc:
                    self.status = str(exc)
                    print(f"Refine failed: {exc}")
            elif key == ord("u"):
                if self.hints:
                    self.hints.pop()
                    self.status = "Removed last click"
                else:
                    self.status = "No clicks to undo"
            elif key == ord("c"):
                self.hints.clear()
                self.mask = None
                self.status = "Cleared clicks and current mask"
            elif key == ord("s"):
                try:
                    self._save()
                except Exception as exc:
                    self.status = str(exc)
                    print(f"Save failed: {exc}")
            elif key in (27, ord("q")):
                break

        cv2.destroyAllWindows()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    rgb_dir = data_dir / "rgb"
    depth_dir = data_dir / "depth"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    rgb_path = resolve_rgb_path(rgb_dir, args.frame)
    image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB frame: {rgb_path}")

    depth_mm = load_depth(depth_dir / rgb_path.name)
    masks_dir = ensure_masks_dir(data_dir)
    mask_path = masks_dir / rgb_path.name

    tool = FirstFrameMaskTool(
        image_bgr=image_bgr,
        depth_mm=depth_mm,
        rgb_path=rgb_path,
        mask_path=mask_path,
        point_radius=args.point_radius,
        depth_tolerance_mm=args.depth_tolerance_mm,
        max_display_size=args.max_display_size,
        use_depth_prior=not args.no_depth_prior,
    )
    tool.run()


if __name__ == "__main__":
    main()
