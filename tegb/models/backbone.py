from __future__ import annotations

import re
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(x, dict):
        for item in x.values():
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


class YOLOFeatureBackboneAdapter(nn.Module):
    """Extract object-level deep features from Ultralytics YOLO via forward hook."""

    def __init__(
        self,
        weights: str,
        target_layer: str,
        crop_size: int = 224,
        device: str = "cuda",
        train_backbone: bool = False,
    ) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError("Ultralytics is required for YOLOFeatureBackboneAdapter.") from e

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.crop_size = int(crop_size)
        self.train_backbone = train_backbone
        self.yolo = YOLO(weights, task="detect")
        self.model = self.yolo.model.to(self.device)
        self.target_layer = target_layer
        self._hook_output = None
        self._hook_handle = None
        self._register_hook()

        if not self.train_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def _register_hook(self) -> None:
        module_map = {name: module for name, module in self.model.named_modules()}

        # 1) Exact match.
        if self.target_layer in module_map:
            self._hook_handle = module_map[self.target_layer].register_forward_hook(self._hook_fn)
            return

        # 2) Common naming variants across Ultralytics versions.
        variants = []
        if self.target_layer.startswith("model.model."):
            variants.append(self.target_layer.replace("model.model.", "model.", 1))
        if self.target_layer.startswith("model."):
            variants.append(self.target_layer.replace("model.", "model.model.", 1))
        for v in variants:
            if v in module_map:
                self._hook_handle = module_map[v].register_forward_hook(self._hook_fn)
                self.target_layer = v
                print(f"[TEGB] target_layer resolved by alias: {v}")
                return

        # 3) Resolve by trailing layer index (e.g. model.model.22 -> index 22 in self.model.model).
        match = re.search(r"(\d+)$", self.target_layer)
        if match is not None and hasattr(self.model, "model"):
            idx = int(match.group(1))
            model_seq = getattr(self.model, "model")
            if hasattr(model_seq, "__len__") and hasattr(model_seq, "__getitem__") and idx < len(model_seq):
                target_module = model_seq[idx]
                for name, module in module_map.items():
                    if module is target_module:
                        self._hook_handle = module.register_forward_hook(self._hook_fn)
                        self.target_layer = name
                        print(f"[TEGB] target_layer resolved by index {idx}: {name}")
                        return
                # Fallback: hook the module object directly even if name was not found.
                self._hook_handle = target_module.register_forward_hook(self._hook_fn)
                self.target_layer = f"<index:{idx}>"
                print(f"[TEGB] target_layer resolved by index {idx} (unnamed module)")
                return

        # 4) Helpful error for config correction.
        sample_names = list(module_map.keys())[:80]
        raise ValueError(
            f"Target layer not found in YOLO model: {self.target_layer}. "
            f"Try names like 'model.<idx>' or check model.named_modules(). "
            f"First available names: {sample_names}"
        )

    def _hook_fn(self, module, inp, out):
        self._hook_output = out

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _crop_resize(self, image: torch.Tensor, box: torch.Tensor) -> torch.Tensor | None:
        # image: [C,H,W], box: [x1,y1,x2,y2] in image coordinates
        c, h, w = image.shape
        x1, y1, x2, y2 = box.tolist()
        x1 = int(max(0, min(w - 1, x1)))
        x2 = int(max(x1 + 1, min(w, x2)))
        y1 = int(max(0, min(h - 1, y1)))
        y2 = int(max(y1 + 1, min(h, y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[:, y1:y2, x1:x2].unsqueeze(0)
        crop = F.interpolate(
            crop,
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        return crop.squeeze(0)

    def _pool_hook_output(self) -> torch.Tensor:
        tensor = _first_tensor(self._hook_output)
        if tensor is None:
            raise RuntimeError("Hook did not capture tensor output.")
        if tensor.ndim == 4:
            return tensor.mean(dim=(-1, -2))
        if tensor.ndim == 3:
            return tensor.mean(dim=-1)
        if tensor.ndim == 2:
            return tensor
        raise ValueError(f"Unsupported hooked tensor shape: {tuple(tensor.shape)}")

    @torch.no_grad()
    def infer_feature_dim(self) -> int:
        x = torch.zeros((1, 3, self.crop_size, self.crop_size), device=self.device)
        self._hook_output = None
        _ = self.model(x)
        feats = self._pool_hook_output()
        return int(feats.shape[-1])

    def extract_features_from_batch(
        self, images: torch.Tensor, boxes_batch: List[torch.Tensor], classes_batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crops = []
        classes = []
        image_idx = []
        for i in range(images.shape[0]):
            img = images[i]
            boxes = boxes_batch[i]
            cls = classes_batch[i]
            if boxes.numel() == 0:
                continue
            for j, box in enumerate(boxes):
                crop = self._crop_resize(img, box)
                if crop is None:
                    continue
                crops.append(crop)
                classes.append(int(cls[j].item()))
                image_idx.append(i)

        if not crops:
            return (
                torch.zeros((0, 1), device=self.device),
                torch.zeros((0,), dtype=torch.long, device=self.device),
                torch.zeros((0,), dtype=torch.long, device=self.device),
            )

        crop_batch = torch.stack(crops, dim=0).to(self.device)
        self._hook_output = None
        if self.train_backbone:
            _ = self.model(crop_batch)
        else:
            with torch.no_grad():
                _ = self.model(crop_batch)
        feats = self._pool_hook_output()
        return (
            feats,
            torch.tensor(classes, dtype=torch.long, device=self.device),
            torch.tensor(image_idx, dtype=torch.long, device=self.device),
        )
