from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tegb.config.schema import DataSection


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_sample_index(images_dir: Path, labels_dir: Path, max_samples: int) -> List[Tuple[Path, Path]]:
    files: List[Tuple[Path, Path]] = []
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(images_dir)
            label = labels_dir / rel.with_suffix(".txt")
            files.append((p, label))
        if max_samples and len(files) >= max_samples:
            break
    return files


class COCOSubsetDataset(Dataset):
    """YOLO-format dataset reader for COCO-style subset experiments."""

    def __init__(
        self,
        data_cfg: DataSection,
        selected_indices: np.ndarray | None = None,
        split_name: str = "all",
        split_seed: int = 42,
        base_samples: List[Tuple[Path, Path]] | None = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.images_dir = Path(data_cfg.images_dir)
        self.labels_dir = Path(data_cfg.labels_dir)
        self.image_size = int(data_cfg.image_size)
        self.split_name = str(split_name)
        self.split_seed = int(split_seed)
        if base_samples is None:
            base_samples = self._build_index(max_samples=data_cfg.max_samples)
        self.total_samples_before_split = len(base_samples)
        if selected_indices is None:
            selected = np.arange(self.total_samples_before_split, dtype=np.int64)
        else:
            selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
            selected = selected[(selected >= 0) & (selected < self.total_samples_before_split)]
        self.selected_indices = selected.astype(np.int64)
        self.samples = [base_samples[int(i)] for i in self.selected_indices.tolist()]
        self.sample_classes, self.class_hist = self._build_label_metadata()

    def _build_index(self, max_samples: int) -> List[Tuple[Path, Path]]:
        return _build_sample_index(self.images_dir, self.labels_dir, max_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_label_classes(self, label_path: Path) -> List[int]:
        if not label_path.exists():
            return []
        classes: List[int] = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    c = int(float(parts[0]))
                except Exception:
                    continue
                classes.append(c)
        return classes

    def _build_label_metadata(self) -> Tuple[List[List[int]], Dict[int, int]]:
        sample_classes: List[List[int]] = []
        class_hist: Dict[int, int] = {}
        for _, label_path in self.samples:
            cls = self._read_label_classes(label_path)
            sample_classes.append(cls)
            for c in cls:
                class_hist[c] = class_hist.get(c, 0) + 1
        return sample_classes, class_hist

    def _read_labels(self, label_path: Path, width: int, height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not label_path.exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )
        boxes = []
        cls = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                c, cx, cy, w, h = parts
                c = int(float(c))
                cx = float(cx) * width
                cy = float(cy) * height
                bw = float(w) * width
                bh = float(h) * height
                x1 = max(0.0, cx - bw / 2.0)
                y1 = max(0.0, cy - bh / 2.0)
                x2 = min(float(width), cx + bw / 2.0)
                y2 = min(float(height), cy + bh / 2.0)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                cls.append(c)
        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(cls, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path, label_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        boxes, classes = self._read_labels(label_path, orig_w, orig_h)
        # Scale boxes to resized resolution.
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= self.image_size / float(orig_w)
            boxes[:, [1, 3]] *= self.image_size / float(orig_h)

        return {
            "image": img_t,
            "boxes": boxes,
            "class_targets": classes,
            "image_id": str(img_path),
        }


def _collate_fn(batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[torch.Tensor] | List[str]]:
    images = torch.stack([x["image"] for x in batch], dim=0)
    boxes = [x["boxes"] for x in batch]
    class_targets = [x["class_targets"] for x in batch]
    image_ids = [x["image_id"] for x in batch]
    return {
        "images": images,
        "boxes": boxes,
        "class_targets": class_targets,
        "image_ids": image_ids,
    }


def _split_indices(num_samples: int, val_split_ratio: float, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    all_idx = np.arange(num_samples, dtype=np.int64)
    if num_samples <= 1:
        return all_idx, all_idx
    if val_split_ratio <= 0.0:
        return all_idx, all_idx

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(num_samples).astype(np.int64)
    n_val = int(round(float(num_samples) * float(val_split_ratio)))
    n_val = max(1, min(num_samples - 1, n_val))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def build_dataloader(data_cfg: DataSection, for_eval: bool = False, split_seed: int = 42) -> DataLoader:
    base_samples = _build_sample_index(
        images_dir=Path(data_cfg.images_dir),
        labels_dir=Path(data_cfg.labels_dir),
        max_samples=int(data_cfg.max_samples),
    )
    train_idx, val_idx = _split_indices(
        num_samples=len(base_samples),
        val_split_ratio=float(data_cfg.val_split_ratio),
        split_seed=int(split_seed),
    )
    selected = val_idx if for_eval else train_idx
    ds = COCOSubsetDataset(
        data_cfg,
        selected_indices=selected,
        split_name=("val" if for_eval else "train"),
        split_seed=split_seed,
        base_samples=base_samples,
    )
    # Long-tail policy:
    # keep the original class distribution for core training/evaluation.
    # Weighted sampler is intentionally disabled even if config enables it.
    sampler = None
    use_shuffle = False
    if not for_eval and sampler is None and bool(data_cfg.shuffle) and len(ds) > 0:
        use_shuffle = True
    return DataLoader(
        ds,
        batch_size=data_cfg.batch_size,
        shuffle=use_shuffle,
        num_workers=data_cfg.num_workers,
        sampler=sampler,
        collate_fn=_collate_fn,
        pin_memory=False,
    )
