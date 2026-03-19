# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_binary_mask(mask_file):
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    if mask.ndim == 3:
        mask = mask.sum(axis=-1) > 0
    else:
        mask = mask > 0
    return mask.astype(np.uint8)


class _PrecomputedMaskBackend:
    def __init__(self):
        self.shorter_side = int(os.environ.get("BUNDLESDF_SEGMENTER_SHORTER_SIDE", "480"))

    def _resize_like_bundlesdf_reader(self, mask, mask_file):
        if self.shorter_side <= 0:
            return mask
        mask_path = Path(mask_file).expanduser().resolve()
        rgb_path = mask_path.parent.parent / "rgb" / mask_path.name
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            return mask
        h, w = image.shape[:2]
        scale = self.shorter_side / min(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    def run(self, mask_file=None):
        mask = _load_binary_mask(mask_file)
        return self._resize_like_bundlesdf_reader(mask, mask_file)


class _XMemBackend:
    def __init__(self):
        self.code_dir = Path(__file__).resolve().parent
        self.xmem_root = Path(
            os.environ.get("BUNDLESDF_XMEM_ROOT", self.code_dir / "BundleTrack" / "XMem")
        ).resolve()
        self.model_path = Path(
            os.environ.get("BUNDLESDF_XMEM_MODEL", self.xmem_root / "saves" / "XMem.pth")
        ).resolve()
        self.size = int(os.environ.get("BUNDLESDF_XMEM_SIZE", "480"))
        self.shorter_side = int(os.environ.get("BUNDLESDF_SEGMENTER_SHORTER_SIDE", str(self.size)))
        self.mem_every = int(os.environ.get("BUNDLESDF_XMEM_MEM_EVERY", "5"))
        self.deep_update_every = int(os.environ.get("BUNDLESDF_XMEM_DEEP_UPDATE_EVERY", "-1"))
        self.disable_long_term = os.environ.get("BUNDLESDF_XMEM_DISABLE_LONG_TERM", "0") == "1"
        self.save_predictions = os.environ.get("BUNDLESDF_XMEM_SAVE", "1") != "0"
        self.save_dir_name = os.environ.get("BUNDLESDF_XMEM_SAVE_DIR", "masks_xmem")
        self.reinit_on_new_mask = os.environ.get("BUNDLESDF_XMEM_REINIT_ON_MASK", "0") == "1"

        if not self.xmem_root.is_dir():
            raise RuntimeError(
                "XMem code not found.\n"
                f"Expected repository at: {self.xmem_root}\n"
                "Please clone https://github.com/hkchengrex/XMem there, or set BUNDLESDF_XMEM_ROOT."
            )
        if not self.model_path.is_file():
            raise RuntimeError(
                "XMem weights not found.\n"
                f"Expected model at: {self.model_path}\n"
                "Download XMem.pth from the official XMem release, or set BUNDLESDF_XMEM_MODEL."
            )

        xmem_root_str = str(self.xmem_root)
        if xmem_root_str not in sys.path:
            sys.path.insert(0, xmem_root_str)

        try:
            import torch
            from inference.inference_core import InferenceCore
            from model.network import XMem
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Failed to import XMem dependencies. "
                "Please run BundleSDF inside the `bundlesdf` environment with torch/torchvision installed."
            ) from exc

        self.torch = torch
        self.F = torch.nn.functional
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            "model": str(self.model_path),
            "benchmark": False,
            "disable_long_term": self.disable_long_term,
            "enable_long_term": not self.disable_long_term,
            "enable_long_term_count_usage": False,
            "max_mid_term_frames": 10,
            "min_mid_term_frames": 5,
            "max_long_term_elements": 10000,
            "num_prototypes": 128,
            "top_k": 30,
            "mem_every": self.mem_every,
            "deep_update_every": self.deep_update_every,
            "size": self.size,
        }

        self.network = XMem(
            self.config,
            model_path=str(self.model_path),
            map_location=self.device,
        ).to(self.device).eval()
        self.processor = InferenceCore(self.network, config=self.config)
        self.processor.set_all_labels([1])

        self.image_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

        self.initialized = False
        self.current_sequence_root = None

    def _mask_path_to_rgb_path(self, mask_path):
        if mask_path.parent.name != "masks":
            raise RuntimeError(
                f"Expected mask path under a masks/ directory, got: {mask_path}"
            )
        return mask_path.parent.parent / "rgb" / mask_path.name

    def _get_save_path(self, mask_path):
        if not self.save_predictions:
            return None
        sequence_root = mask_path.parent.parent
        save_dir = sequence_root / self.save_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / mask_path.name

    def _prepare_image(self, rgb_path):
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"RGB frame not found: {rgb_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.size > 0:
            orig_h, orig_w = image_rgb.shape[:2]
            min_hw = min(orig_h, orig_w)
            if min_hw != self.size:
                new_h = int(round(orig_h / min_hw * self.size))
                new_w = int(round(orig_w / min_hw * self.size))
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        image_tensor = self.torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        image_tensor = (image_tensor - self.image_mean) / self.image_std

        return image_tensor

    def _prepare_mask(self, mask_path, shape):
        mask = _load_binary_mask(mask_path).astype(np.float32)
        target_h, target_w = shape
        if mask.shape != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_tensor = self.torch.from_numpy(mask).to(self.device)
        return mask_tensor.unsqueeze(0)

    def _predict_mask(self, image_tensor, mask_tensor):
        with self.torch.inference_mode():
            valid_labels = [1] if mask_tensor is not None else None
            prob = self.processor.step(image_tensor, mask_tensor, valid_labels, end=False)
            out_mask = self.torch.argmax(prob, dim=0).detach().cpu().numpy().astype(np.uint8)
        return (out_mask > 0).astype(np.uint8)

    def _maybe_reset_for_new_sequence(self, mask_path):
        sequence_root = mask_path.parent.parent.resolve()
        if self.current_sequence_root != sequence_root:
            self.processor.clear_memory()
            self.processor.set_all_labels([1])
            self.initialized = False
            self.current_sequence_root = sequence_root

    def run(self, mask_file=None):
        mask_path = Path(mask_file).expanduser().resolve()
        rgb_path = self._mask_path_to_rgb_path(mask_path)

        self._maybe_reset_for_new_sequence(mask_path)
        image_tensor = self._prepare_image(rgb_path)

        use_input_mask = False
        if not self.initialized:
            use_input_mask = True
        elif self.reinit_on_new_mask and mask_path.is_file():
            use_input_mask = True

        mask_tensor = None
        if use_input_mask:
            if not mask_path.is_file():
                raise RuntimeError(
                    "XMem needs an initial object mask to start propagation.\n"
                    f"Missing mask: {mask_path}\n"
                    "Please provide at least the first-frame object mask in masks/000000.png."
                )
            mask_tensor = self._prepare_mask(mask_path, image_tensor.shape[-2:])

        pred_mask = self._predict_mask(image_tensor, mask_tensor)
        self.initialized = True

        save_path = self._get_save_path(mask_path)
        if save_path is not None:
            cv2.imwrite(str(save_path), pred_mask.astype(np.uint8) * 255)

        return pred_mask


class Segmenter:
    def __init__(self):
        mode = os.environ.get("BUNDLESDF_SEGMENTER_MODE", "xmem").strip().lower()
        if mode == "precomputed":
            self.backend = _PrecomputedMaskBackend()
        elif mode == "xmem":
            self.backend = _XMemBackend()
        else:
            raise ValueError(
                f"Unsupported segmenter mode: {mode}. "
                "Use BUNDLESDF_SEGMENTER_MODE=xmem or precomputed."
            )

    def run(self, mask_file=None):
        return self.backend.run(mask_file)
