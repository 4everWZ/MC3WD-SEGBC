from __future__ import annotations

import gc
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

from tegb.config.schema import ExperimentConfig
from tegb.data import build_dataloader
from tegb.decision import build_decider
from tegb.granular import build_granular_builder
from tegb.losses import (
    DetectionSurrogateLoss,
    EvidentialLoss,
    GromovWassersteinLoss,
    HardNegativeInfoNCELoss,
    PersistentTopologyLoss,
)
from tegb.metrics import compute_cluster_indices, compute_coverage_at_risk, compute_fpr95, compute_vsf_report
from tegb.metrics import compute_manifold_diagnostics
from tegb.models import EvidentialHead, VisProjectionHead, YOLOFeatureBackboneAdapter
from tegb.runner.visualization import export_embedding_projection_comparison, export_visualization_artifacts


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _scalar(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _progress(iterable, total: int | None = None, desc: str = "", leave: bool = False):
    if _tqdm is None:
        return iterable
    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)() or getattr(sys.stderr, "isatty", lambda: False)())
    return _tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True, disable=not is_tty)


def _progress_note(message: str) -> None:
    text = str(message).strip()
    if not text:
        return
    if _tqdm is not None:
        try:
            _tqdm.write(text)
            return
        except Exception:
            pass
    print(text)


class ResearchTrainer:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.experiment.seed)
        self.device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
        self.run_dir = cfg.run_dir
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

        self.train_loader = build_dataloader(cfg.data, for_eval=False, split_seed=cfg.experiment.seed)
        self.eval_loader = build_dataloader(cfg.data, for_eval=True, split_seed=cfg.experiment.seed)
        self.last_eval_extras: Dict[str, float] = {}
        self._write_data_split_snapshot()

        self.backbone = YOLOFeatureBackboneAdapter(
            weights=cfg.experiment.weights,
            target_layer=cfg.experiment.target_layer,
            crop_size=cfg.data.crop_size,
            device=str(self.device),
            train_backbone=cfg.optim.train_backbone,
        )
        feat_dim = self.backbone.infer_feature_dim()
        self.evidential_head = EvidentialHead(feat_dim, cfg.experiment.num_classes).to(self.device)
        self.proj_head = VisProjectionHead(feat_dim, hidden_dim=256, out_dim=2).to(self.device)

        params = list(self.evidential_head.parameters()) + list(self.proj_head.parameters())
        if cfg.optim.train_backbone:
            params += list(self.backbone.model.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )

        self.det_class_weights = self._build_detection_class_weights()
        self.det_loss_fn = DetectionSurrogateLoss(class_weights=self.det_class_weights)
        self.edl_loss_fn = EvidentialLoss()
        self.gw_loss_fn = GromovWassersteinLoss(
            entropy_reg=cfg.loss.gw_entropy_reg,
            metric_weight=cfg.loss.gw_metric_weight,
        )
        self.topo_loss_fn = PersistentTopologyLoss(
            max_points=cfg.loss.topo_max_points,
            persistence_threshold=cfg.granular.ph_persistence_threshold,
        )
        self.hn_loss_fn = HardNegativeInfoNCELoss()
        self.decider = build_decider(cfg.decision)
        self.granular_builder = build_granular_builder(cfg.granular, random_state=cfg.experiment.seed)
        self._best_metric_name = str(cfg.experiment.best_metric)
        self._best_mode = str(cfg.experiment.best_mode)
        self._best_value = -np.inf if self._best_mode == "max" else np.inf
        self._best_epoch = 0
        self._best_found = False

    def close(self) -> None:
        for loader_name in ("train_loader", "eval_loader"):
            loader = getattr(self, loader_name, None)
            if loader is None:
                continue
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:
                        pass
            setattr(self, loader_name, None)
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _write_data_split_snapshot(self) -> None:
        train_ds = getattr(self.train_loader, "dataset", None)
        val_ds = getattr(self.eval_loader, "dataset", None)
        payload = {
            "seed": int(self.cfg.experiment.seed),
            "val_split_ratio": float(self.cfg.data.val_split_ratio),
            "train_count": int(len(train_ds)) if train_ds is not None else 0,
            "val_count": int(len(val_ds)) if val_ds is not None else 0,
            "total_before_split": int(getattr(train_ds, "total_samples_before_split", 0)) if train_ds is not None else 0,
            "train_split_name": str(getattr(train_ds, "split_name", "train")) if train_ds is not None else "train",
            "val_split_name": str(getattr(val_ds, "split_name", "val")) if val_ds is not None else "val",
        }
        path = self.run_dir / "data_split.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _build_detection_class_weights(self) -> torch.Tensor | None:
        if not self.cfg.loss.enable_class_balanced_det:
            return None
        dataset = getattr(self.train_loader, "dataset", None)
        class_hist = getattr(dataset, "class_hist", None)
        if not isinstance(class_hist, dict) or not class_hist:
            return None

        num_classes = int(self.cfg.experiment.num_classes)
        counts = np.zeros((num_classes,), dtype=np.float64)
        for k, v in class_hist.items():
            c = int(k)
            if 0 <= c < num_classes:
                counts[c] += float(v)

        seen = counts > 0
        if not np.any(seen):
            return None

        power = float(max(self.cfg.loss.det_class_balance_power, 0.0))
        weights = np.ones_like(counts, dtype=np.float64)
        if power > 0.0:
            weights[seen] = (np.sum(counts[seen]) / np.maximum(counts[seen], 1.0)) ** power

        weights = np.clip(weights, self.cfg.loss.det_class_weight_min, self.cfg.loss.det_class_weight_max)
        norm = float(np.mean(weights[seen])) if np.any(seen) else 1.0
        if norm > 1e-12:
            weights = weights / norm
        return torch.from_numpy(weights.astype(np.float32))

    def _resolve_class_name_map(self) -> Dict[int, str]:
        names = None
        yolo = getattr(self.backbone, "yolo", None)
        if yolo is not None:
            names = getattr(yolo, "names", None)

        class_map: Dict[int, str] = {}
        if isinstance(names, dict):
            for k, v in names.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                class_map[idx] = str(v)
        elif isinstance(names, (list, tuple)):
            for idx, v in enumerate(names):
                class_map[idx] = str(v)

        if not class_map:
            for i in range(int(self.cfg.experiment.num_classes)):
                class_map[i] = f"class_{i}"
        return class_map

    def _forward_batch(self, batch) -> Dict[str, torch.Tensor]:
        images = batch["images"].to(self.device)
        boxes = [b.to(self.device) for b in batch["boxes"]]
        cls = [c.to(self.device) for c in batch["class_targets"]]

        feats, targets, _ = self.backbone.extract_features_from_batch(images, boxes, cls)
        if feats.numel() == 0 or feats.shape[0] == 0:
            return {}
        dir_out = self.evidential_head(feats)
        low = self.proj_head(feats)
        return {
            "features": feats,
            "targets": targets,
            "low": low,
            "dir_alpha": dir_out.alpha,
            "dir_evidence": dir_out.evidence,
            "dir_uncertainty": dir_out.uncertainty,
        }

    def _compute_losses(self, fw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from tegb.types import DirichletOutput

        out = DirichletOutput(
            alpha=fw["dir_alpha"],
            evidence=fw["dir_evidence"],
            uncertainty=fw["dir_uncertainty"],
        )

        l_det = self.det_loss_fn(out, fw["targets"])
        l_edl = self.edl_loss_fn(out, fw["targets"]) if self.cfg.loss.enable_edl else torch.zeros((), device=self.device)
        l_gw = (
            self.gw_loss_fn(fw["features"], fw["low"]) if self.cfg.loss.enable_gw else torch.zeros((), device=self.device)
        )
        l_ph = self.topo_loss_fn(fw["features"]) if self.cfg.loss.enable_ph else torch.zeros((), device=self.device)
        l_hn = self.hn_loss_fn(fw["features"], fw["targets"])

        total = l_det
        total = total + self.cfg.loss.lambda_edl * l_edl
        total = total + self.cfg.loss.lambda_gw * l_gw
        total = total + self.cfg.loss.lambda_ph * l_ph
        total = total + self.cfg.loss.lambda_hn * l_hn
        if not torch.isfinite(total):
            # Automatic fallback for unstable steps.
            total = l_det + 0.1 * torch.nan_to_num(l_edl) + 0.1 * torch.nan_to_num(l_hn)
        return {
            "total": total,
            "det": l_det,
            "edl": l_edl,
            "gw": l_gw,
            "ph": l_ph,
            "hn": l_hn,
        }

    def fit(self) -> Dict[str, float]:
        history: List[Dict[str, float]] = []
        epoch_iter = _progress(
            range(1, self.cfg.optim.epochs + 1),
            total=self.cfg.optim.epochs,
            desc="TEGB Train Epochs",
            leave=True,
        )
        for epoch in epoch_iter:
            self.evidential_head.train()
            self.proj_head.train()
            epoch_losses = {"total": [], "det": [], "edl": [], "gw": [], "ph": [], "hn": []}

            train_iter = _progress(
                self.train_loader,
                total=(len(self.train_loader) if hasattr(self.train_loader, "__len__") else None),
                desc=f"Epoch {epoch}/{self.cfg.optim.epochs} Train",
                leave=False,
            )
            for batch in train_iter:
                fw = self._forward_batch(batch)
                if not fw:
                    continue
                losses = self._compute_losses(fw)

                self.optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(
                    list(self.evidential_head.parameters()) + list(self.proj_head.parameters()),
                    max_norm=self.cfg.optim.grad_clip,
                )
                self.optimizer.step()

                for k in epoch_losses:
                    epoch_losses[k].append(_scalar(losses[k]))

            metrics = self._evaluate_losses(progress_desc=f"Epoch {epoch}/{self.cfg.optim.epochs} Eval(loss)")
            row = {
                "epoch": float(epoch),
                "train_total": float(np.mean(epoch_losses["total"])) if epoch_losses["total"] else 0.0,
                "train_det": float(np.mean(epoch_losses["det"])) if epoch_losses["det"] else 0.0,
                "train_edl": float(np.mean(epoch_losses["edl"])) if epoch_losses["edl"] else 0.0,
                "train_gw": float(np.mean(epoch_losses["gw"])) if epoch_losses["gw"] else 0.0,
                "train_ph": float(np.mean(epoch_losses["ph"])) if epoch_losses["ph"] else 0.0,
                "train_hn": float(np.mean(epoch_losses["hn"])) if epoch_losses["hn"] else 0.0,
            }
            row.update(metrics)
            history.append(row)
            self._maybe_update_best(epoch=epoch, row=row)

            if epoch % self.cfg.experiment.checkpoint_every == 0:
                self._save_checkpoint(epoch)

        if self.cfg.experiment.save_best and not self._best_found and history:
            last = history[-1]
            fallback = float(last.get("train_total", np.nan))
            if np.isfinite(fallback):
                self._best_found = True
                self._best_epoch = int(last.get("epoch", self.cfg.optim.epochs))
                self._best_value = fallback
                self._best_metric_name = f"{self.cfg.experiment.best_metric}->train_total_fallback"
                self._save_checkpoint(self._best_epoch, name="best.pt")
                self._write_best_summary()

        self._save_checkpoint(self.cfg.optim.epochs, name="last.pt")

        final_eval: Dict[str, float] = {}
        best_ckpt = self.ckpt_dir / "best.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location="cpu")
            self.evidential_head.load_state_dict(ckpt["evidential_head"])
            self.proj_head.load_state_dict(ckpt["proj_head"])
        final_eval = self.evaluate(progress_desc="TEGB Final Eval (best/last)", export_artifacts=True)
        if self._best_found:
            self.last_eval_extras["best_epoch"] = float(self._best_epoch)
            self.last_eval_extras["best_metric_value"] = float(self._best_value)
            self.last_eval_extras["best_mode_is_max"] = 1.0 if self._best_mode == "max" else 0.0
            final_eval["best_epoch"] = float(self._best_epoch)
            final_eval["best_metric_value"] = float(self._best_value)
        with (self.run_dir / "best_eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(final_eval, f, ensure_ascii=False, indent=2)

        self._write_history(history, extras=self.last_eval_extras)
        out = dict(history[-1]) if history else {}
        out.update(final_eval)
        return out

    def _is_better(self, value: float, mode: str | None = None) -> bool:
        if not np.isfinite(value):
            return False
        if not self._best_found:
            return True
        compare_mode = (mode or self._best_mode)
        if compare_mode == "max":
            return value > self._best_value
        return value < self._best_value

    def _write_best_summary(self) -> None:
        path = self.run_dir / "best_summary.json"
        payload = {
            "metric": self._best_metric_name,
            "mode": self._best_mode,
            "epoch": int(self._best_epoch),
            "value": float(self._best_value),
            "checkpoint": str(self.ckpt_dir / "best.pt"),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _maybe_update_best(self, epoch: int, row: Dict[str, float]) -> None:
        if not self.cfg.experiment.save_best:
            return
        key = str(self.cfg.experiment.best_metric)
        mode = str(self._best_mode)
        metric_name = key
        if key not in row:
            if "val_total" in row:
                key = "val_total"
                mode = "min"
                metric_name = f"{self.cfg.experiment.best_metric}->val_total_fallback"
            elif "train_total" in row:
                key = "train_total"
                mode = "min"
                metric_name = f"{self.cfg.experiment.best_metric}->train_total_fallback"
            else:
                return
        value = float(row.get(key, np.nan))
        if not self._is_better(value, mode=mode):
            return
        self._best_found = True
        self._best_value = value
        self._best_epoch = int(epoch)
        self._best_metric_name = metric_name
        self._best_mode = mode
        self._save_checkpoint(epoch, name="best.pt")
        self._write_best_summary()

    def _save_checkpoint(self, epoch: int, name: str | None = None) -> None:
        ckpt_name = name or f"epoch_{epoch}.pt"
        path = self.ckpt_dir / ckpt_name
        torch.save(
            {
                "epoch": epoch,
                "cfg": self.cfg.to_dict(),
                "evidential_head": self.evidential_head.state_dict(),
                "proj_head": self.proj_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def _write_history(self, history: List[Dict[str, float]], extras: Dict[str, float] | None = None) -> None:
        with (self.run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"history": history, "extras": extras or {}}, f, ensure_ascii=False, indent=2)

    def _subsample_indices(self, n: int, max_points: int, seed_offset: int = 0) -> np.ndarray:
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        if max_points <= 0 or n <= max_points:
            return np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(int(self.cfg.experiment.seed) + int(seed_offset))
        idx = rng.choice(n, size=max_points, replace=False)
        return np.sort(idx.astype(np.int64))

    def _cuda_memory_mb(self) -> tuple[float, float]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0.0, 0.0
        device_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
        allocated = float(torch.cuda.memory_allocated(device_idx) / (1024.0 * 1024.0))
        reserved = float(torch.cuda.memory_reserved(device_idx) / (1024.0 * 1024.0))
        return allocated, reserved

    def _chunked_support_count(
        self,
        features: np.ndarray,
        centers: np.ndarray,
        radii: np.ndarray,
        target_cells: int = 2_000_000,
    ) -> np.ndarray:
        if features.ndim != 2 or centers.ndim != 2 or centers.shape[0] == 0:
            return np.zeros((features.shape[0],), dtype=np.int64)
        chunk = max(64, int(target_cells // max(1, int(centers.shape[0]))))
        out = np.zeros((features.shape[0],), dtype=np.int64)
        for start in range(0, features.shape[0], chunk):
            end = min(features.shape[0], start + chunk)
            feat_chunk = np.asarray(features[start:end], dtype=np.float32, copy=False)
            ctr = np.asarray(centers, dtype=np.float32, copy=False)
            rr = np.asarray(radii, dtype=np.float32, copy=False)
            dist = np.linalg.norm(feat_chunk[:, None, :] - ctr[None, :, :], axis=2)
            out[start:end] = np.sum(dist <= rr[None, :], axis=1).astype(np.int64)
        return out

    def _save_npy_chunked(self, path: Path, arr: np.ndarray, chunk_size: int = 8192) -> None:
        arr_np = np.asarray(arr)
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=arr_np.dtype, shape=arr_np.shape)
        if arr_np.ndim == 0:
            mm[...] = arr_np
        elif arr_np.shape[0] == 0:
            pass
        else:
            n = arr_np.shape[0]
            for s in range(0, n, chunk_size):
                e = min(n, s + chunk_size)
                mm[s:e] = arr_np[s:e]
        del mm

    @torch.no_grad()
    def _collect_eval_arrays(self, progress_desc: str | None = None, stream_to_disk: bool = False):
        if not stream_to_disk:
            all_high = []
            all_low = []
            all_alpha = []
            all_targets = []
            all_unc = []
            eval_iter = _progress(
                self.eval_loader,
                total=(len(self.eval_loader) if hasattr(self.eval_loader, "__len__") else None),
                desc=(progress_desc or "TEGB Eval"),
                leave=False,
            )
            for batch in eval_iter:
                fw = self._forward_batch(batch)
                if not fw:
                    continue
                alpha = fw["dir_alpha"]
                all_high.append(fw["features"].detach().cpu().numpy())
                all_low.append(fw["low"].detach().cpu().numpy())
                all_alpha.append(alpha.detach().cpu().numpy())
                all_targets.append(fw["targets"].detach().cpu().numpy())
                all_unc.append(fw["dir_uncertainty"].detach().cpu().numpy())
            if not all_high:
                return None
            return {
                "high": np.concatenate(all_high, axis=0),
                "low": np.concatenate(all_low, axis=0),
                "alpha": np.concatenate(all_alpha, axis=0),
                "targets": np.concatenate(all_targets, axis=0),
                "uncertainty": np.concatenate(all_unc, axis=0),
                "cache_dir": None,
            }

        cache_dir = self.run_dir / "artifacts" / "_eval_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        feat_dim = None
        low_dim = None
        alpha_dim = None

        count_iter = _progress(
            self.eval_loader,
            total=(len(self.eval_loader) if hasattr(self.eval_loader, "__len__") else None),
            desc=((progress_desc or "TEGB Eval") + " [count]"),
            leave=False,
        )
        for batch in count_iter:
            fw = self._forward_batch(batch)
            if not fw:
                continue
            n = int(fw["features"].shape[0])
            if n <= 0:
                continue
            total_rows += n
            if feat_dim is None:
                feat_dim = int(fw["features"].shape[1])
                low_dim = int(fw["low"].shape[1])
                alpha_dim = int(fw["dir_alpha"].shape[1])

        if total_rows <= 0 or feat_dim is None or low_dim is None or alpha_dim is None:
            shutil.rmtree(cache_dir, ignore_errors=True)
            return None

        high_mm = np.lib.format.open_memmap(
            cache_dir / "high.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, feat_dim),
        )
        low_mm = np.lib.format.open_memmap(
            cache_dir / "low.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, low_dim),
        )
        alpha_mm = np.lib.format.open_memmap(
            cache_dir / "alpha.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, alpha_dim),
        )
        target_mm = np.lib.format.open_memmap(
            cache_dir / "targets.npy",
            mode="w+",
            dtype=np.int64,
            shape=(total_rows,),
        )
        unc_mm = np.lib.format.open_memmap(
            cache_dir / "uncertainty.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_rows,),
        )

        fill_iter = _progress(
            self.eval_loader,
            total=(len(self.eval_loader) if hasattr(self.eval_loader, "__len__") else None),
            desc=((progress_desc or "TEGB Eval") + " [materialize]"),
            leave=False,
        )
        cursor = 0
        for batch in fill_iter:
            fw = self._forward_batch(batch)
            if not fw:
                continue
            n = int(fw["features"].shape[0])
            if n <= 0:
                continue
            high_mm[cursor:cursor + n] = fw["features"].detach().cpu().numpy().astype(np.float32, copy=False)
            low_mm[cursor:cursor + n] = fw["low"].detach().cpu().numpy().astype(np.float32, copy=False)
            alpha_mm[cursor:cursor + n] = fw["dir_alpha"].detach().cpu().numpy().astype(np.float32, copy=False)
            target_mm[cursor:cursor + n] = fw["targets"].detach().cpu().numpy().astype(np.int64, copy=False)
            unc_mm[cursor:cursor + n] = fw["dir_uncertainty"].detach().cpu().numpy().astype(np.float32, copy=False)
            cursor += n

        del high_mm, low_mm, alpha_mm, target_mm, unc_mm
        return {
            "high": np.load(cache_dir / "high.npy", mmap_mode="r"),
            "low": np.load(cache_dir / "low.npy", mmap_mode="r"),
            "alpha": np.load(cache_dir / "alpha.npy", mmap_mode="r"),
            "targets": np.load(cache_dir / "targets.npy", mmap_mode="r"),
            "uncertainty": np.load(cache_dir / "uncertainty.npy", mmap_mode="r"),
            "cache_dir": cache_dir,
        }

    @torch.no_grad()
    def _evaluate_losses(self, progress_desc: str | None = None) -> Dict[str, float]:
        self.evidential_head.eval()
        self.proj_head.eval()
        losses_acc = {"total": [], "det": [], "edl": [], "gw": [], "ph": [], "hn": []}
        eval_iter = _progress(
            self.eval_loader,
            total=(len(self.eval_loader) if hasattr(self.eval_loader, "__len__") else None),
            desc=(progress_desc or "TEGB Eval Loss"),
            leave=False,
        )
        total_instances = 0
        for batch in eval_iter:
            fw = self._forward_batch(batch)
            if not fw:
                continue
            step = self._compute_losses(fw)
            for k in losses_acc:
                losses_acc[k].append(_scalar(step[k]))
            total_instances += int(fw["targets"].shape[0])

        out = {
            "val_total": float(np.mean(losses_acc["total"])) if losses_acc["total"] else np.nan,
            "val_det": float(np.mean(losses_acc["det"])) if losses_acc["det"] else np.nan,
            "val_edl": float(np.mean(losses_acc["edl"])) if losses_acc["edl"] else np.nan,
            "val_gw": float(np.mean(losses_acc["gw"])) if losses_acc["gw"] else np.nan,
            "val_ph": float(np.mean(losses_acc["ph"])) if losses_acc["ph"] else np.nan,
            "val_hn": float(np.mean(losses_acc["hn"])) if losses_acc["hn"] else np.nan,
            "val_instances": float(total_instances),
        }
        if not np.isfinite(out["val_total"]):
            out["val_total"] = float(out["val_det"]) if np.isfinite(out["val_det"]) else float("nan")
        return out

    @torch.no_grad()
    def evaluate(
        self,
        progress_desc: str | None = None,
        export_artifacts: bool = True,
        stream_to_disk: bool = False,
    ) -> Dict[str, float]:
        self.evidential_head.eval()
        self.proj_head.eval()
        _progress_note(f"[Eval] start: {progress_desc or 'TEGB Eval'}")
        arrays = self._collect_eval_arrays(progress_desc=progress_desc, stream_to_disk=stream_to_disk)
        if arrays is None:
            return {}

        _progress_note("[Eval] concatenate feature buffers")
        high = arrays["high"]
        low = arrays["low"]
        alpha_all = arrays["alpha"]
        targets = arrays["targets"]
        uncertainty = arrays["uncertainty"]
        probs = alpha_all / np.clip(np.sum(alpha_all, axis=-1, keepdims=True), a_min=1e-8, a_max=None)
        targets = np.asarray(targets, dtype=np.int64)
        uncertainty = np.asarray(uncertainty, dtype=np.float32)
        preds = np.argmax(probs, axis=-1).astype(np.int64, copy=False)
        cuda_alloc_before, cuda_reserved_before = self._cuda_memory_mb()

        # Granular balls in high-dimensional space.
        builder_exception = 0.0
        if self.cfg.loss.enable_tgb:
            _progress_note("[Eval] build granular balls")
            try:
                balls = self.granular_builder.build(
                    high,
                    probs,
                    preds if preds.size else np.zeros((high.shape[0],), dtype=int),
                )
            except Exception:
                balls = []
                builder_exception = 1.0
        else:
            balls = []

        if export_artifacts:
            granular_rows = []
            for b in balls:
                granular_rows.append(
                    {
                        "center": np.asarray(b.center).tolist(),
                        "radius": float(b.radius),
                        "purity": float(b.purity),
                        "members": len(b.members),
                        "topo_state": b.topo_state,
                        "semantic_entropy": b.semantic_entropy,
                        "dominant_class": b.dominant_class,
                        "boundary_score": b.boundary_score,
                        "geometry_type": b.geometry_type,
                        "support_count": b.support_count,
                        "covariance": (np.asarray(b.covariance).tolist() if b.covariance is not None else None),
                        "chi2_threshold": b.chi2_threshold,
                        "confidence_level": b.confidence_level,
                    }
                )
            with (self.run_dir / "artifacts" / "granular_balls.json").open("w", encoding="utf-8") as f:
                json.dump(granular_rows, f, ensure_ascii=False, indent=2)

            if self.cfg.granular.mode == "prob_ellipsoid":
                with (self.run_dir / "artifacts" / "prob_ellipsoids.json").open("w", encoding="utf-8") as f:
                    json.dump(granular_rows, f, ensure_ascii=False, indent=2)
            if self.cfg.granular.mode == "v3_spherical_entropy":
                with (self.run_dir / "artifacts" / "v3_spherical_balls.json").open("w", encoding="utf-8") as f:
                    json.dump(granular_rows, f, ensure_ascii=False, indent=2)

        vsf_idx = self._subsample_indices(high.shape[0], int(self.cfg.vsf.max_points), seed_offset=101)
        high_vsf = high[vsf_idx]
        low_vsf = low[vsf_idx]
        probs_vsf = probs[vsf_idx]
        _progress_note("[Eval] compute VSF/GTF/SNH")
        vsf = compute_vsf_report(
            high=high_vsf,
            low=low_vsf,
            probs=probs_vsf,
            semantic_weight=self.cfg.vsf.semantic_weight,
            ks=list(range(self.cfg.vsf.k_min, self.cfg.vsf.k_max + 1, self.cfg.vsf.k_step)),
        )
        if export_artifacts:
            with (self.run_dir / "vsf_report.json").open("w", encoding="utf-8") as f:
                json.dump(asdict(vsf), f, ensure_ascii=False, indent=2)

        from tegb.types import DirichletOutput

        uncertainty_tensor = torch.from_numpy(np.array(uncertainty, dtype=np.float32, copy=True))

        if self.cfg.decision.mode == "v3_spherical_collision":
            n_points = int(high.shape[0])
            out = DirichletOutput(
                alpha=torch.ones((n_points, 1), dtype=torch.float32),
                evidence=torch.zeros((n_points, 1), dtype=torch.float32),
                uncertainty=uncertainty_tensor,
            )
        else:
            alpha_clipped = np.clip(alpha_all, 1e-8, None)
            alpha_tensor = torch.from_numpy(np.array(alpha_clipped, dtype=np.float32, copy=True))
            out = DirichletOutput(
                alpha=alpha_tensor,
                evidence=torch.zeros_like(alpha_tensor),
                uncertainty=uncertainty_tensor,
            )
        if self.cfg.loss.enable_3wd:
            _progress_note("[Eval] assign three-way regions")
            three = self.decider.decide(out, features=high, balls=balls)
            regions = three.region_labels.cpu().numpy()
            if three.collision_mask is not None:
                collision_mask_np = three.collision_mask.cpu().numpy()
            else:
                collision_mask_np = np.zeros((regions.shape[0],), dtype=bool)
            collision_pairs: List[Dict[str, Any]] = getattr(self.decider, "last_collision_pairs", [])
            if export_artifacts:
                self._save_npy_chunked(self.run_dir / "artifacts" / "collision_mask.npy", collision_mask_np)
                with (self.run_dir / "artifacts" / "collision_pairs.json").open("w", encoding="utf-8") as f:
                    json.dump(collision_pairs, f, ensure_ascii=False, indent=2)
                if three.nearest_ball_index is not None:
                    self._save_npy_chunked(
                        self.run_dir / "artifacts" / "nearest_ball_index.npy",
                        three.nearest_ball_index.cpu().numpy(),
                    )
                else:
                    self._save_npy_chunked(
                        self.run_dir / "artifacts" / "nearest_ball_index.npy",
                        np.full((regions.shape[0],), -1, dtype=np.int64),
                    )
                if three.mahalanobis_score is not None:
                    self._save_npy_chunked(
                        self.run_dir / "artifacts" / "mahalanobis_score.npy",
                        three.mahalanobis_score.cpu().numpy(),
                    )
                else:
                    self._save_npy_chunked(
                        self.run_dir / "artifacts" / "mahalanobis_score.npy",
                        np.full((regions.shape[0],), np.inf, dtype=np.float32),
                    )
        else:
            regions = np.full((probs.shape[0],), fill_value=1, dtype=np.int64)
            collision_mask_np = np.zeros((regions.shape[0],), dtype=bool)
            collision_pairs = []
            if export_artifacts:
                self._save_npy_chunked(self.run_dir / "artifacts" / "collision_mask.npy", collision_mask_np)
                with (self.run_dir / "artifacts" / "collision_pairs.json").open("w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                self._save_npy_chunked(
                    self.run_dir / "artifacts" / "nearest_ball_index.npy",
                    np.full((regions.shape[0],), -1, dtype=np.int64),
                )
                self._save_npy_chunked(
                    self.run_dir / "artifacts" / "mahalanobis_score.npy",
                    np.full((regions.shape[0],), np.inf, dtype=np.float32),
                )
        if export_artifacts:
            _progress_note("[Eval] export base arrays")
            self._save_npy_chunked(self.run_dir / "three_way_regions.npy", regions)
            self._save_npy_chunked(self.run_dir / "embeddings_2d.npy", low)
        vis_cfg = self.cfg.visualization
        if export_artifacts and vis_cfg.enabled:
            _progress_note("[Eval] export visualization figures")
            vis_export_kwargs = {
                "low": low,
                "high": high,
                "preds": preds,
                "targets": targets,
                "regions": regions,
                "uncertainty": uncertainty,
                "collision_mask": collision_mask_np,
                "balls": balls,
                "class_names": self._resolve_class_name_map(),
                "class_groups": vis_cfg.class_groups,
                "include_classes": vis_cfg.include_classes,
                "group_cmap_name": vis_cfg.group_cmap_name,
                "class_legend_max_items": vis_cfg.class_legend_max_items,
                "show_zero_count_classes_in_legend": vis_cfg.show_zero_count_classes_in_legend,
                "focus_label_source": vis_cfg.focus_label_source,
                "class_compare_mode": vis_cfg.class_compare_mode,
                "class_compare_topk": vis_cfg.class_compare_topk,
                "class_compare_label_source": vis_cfg.class_compare_label_source,
                "embedding_source": vis_cfg.embedding_source,
                "embedding_auto_tsne_max_points": vis_cfg.embedding_auto_tsne_max_points,
                "embedding_metric": vis_cfg.embedding_metric,
                "highdim_pre_reduce": vis_cfg.highdim_pre_reduce,
                "highdim_pca_components": vis_cfg.highdim_pca_components,
                "tsne_perplexity": vis_cfg.tsne_perplexity,
                "tsne_max_iter": vis_cfg.tsne_max_iter,
                "umap_n_neighbors": vis_cfg.umap_n_neighbors,
                "umap_min_dist": vis_cfg.umap_min_dist,
                "datamap_enabled": vis_cfg.datamap_enabled,
                "datamap_force_matplotlib": vis_cfg.datamap_force_matplotlib,
                "datamap_use_medoids": vis_cfg.datamap_use_medoids,
                "datamap_point_size": vis_cfg.datamap_point_size,
                "datamap_alpha": vis_cfg.datamap_alpha,
                "datamap_min_font_size": vis_cfg.datamap_min_font_size,
                "datamap_max_font_size": vis_cfg.datamap_max_font_size,
                "datamap_label_wrap_width": vis_cfg.datamap_label_wrap_width,
                "datamap_title_font_size": vis_cfg.datamap_title_font_size,
                "embedding_scatter_alpha": vis_cfg.embedding_scatter_alpha,
                "embedding_color_lighten": vis_cfg.embedding_color_lighten,
                "embedding_pred_dominant_ratio_threshold": vis_cfg.embedding_pred_dominant_ratio_threshold,
                "embedding_pred_dominant_alpha": vis_cfg.embedding_pred_dominant_alpha,
                "balls_3d_enabled": vis_cfg.balls_3d_enabled,
                "balls_3d_radius_quantiles": vis_cfg.balls_3d_radius_quantiles,
                "ball_label_source": vis_cfg.ball_label_source,
                "seed": self.cfg.experiment.seed,
                "max_points": vis_cfg.max_points,
                "embedding_random_subset_per_class": vis_cfg.embedding_random_subset_per_class,
                "max_ball_overlays": vis_cfg.max_ball_overlays,
            }
            vis_meta = export_visualization_artifacts(
                run_dir=self.run_dir,
                **vis_export_kwargs,
            )
            compare_meta = export_embedding_projection_comparison(
                run_dir=self.run_dir,
                embedding_compare_sources=vis_cfg.embedding_compare_sources,
                export_kwargs=vis_export_kwargs,
            )
            vis_meta["embedding_compare_sources"] = list(vis_cfg.embedding_compare_sources)
            vis_meta["embedding_compare_count"] = int(len(compare_meta.get("rows", [])))
            if compare_meta.get("summary_json_path"):
                vis_meta["embedding_compare_summary_path"] = str(compare_meta.get("summary_json_path"))
            if compare_meta.get("summary_csv_path"):
                vis_meta["embedding_compare_summary_csv_path"] = str(compare_meta.get("summary_csv_path"))
            if compare_meta.get("warnings"):
                vis_meta.setdefault("warnings", []).extend(compare_meta.get("warnings", []))
        else:
            vis_meta = {
                "enabled": False,
                "files": [],
                "warnings": (["disabled_by_config"] if vis_cfg.enabled is False else ["artifact_export_skipped"]),
                "num_points_total": int(low.shape[0]),
                "embedding_compare_sources": list(vis_cfg.embedding_compare_sources),
                "embedding_compare_count": 0,
            }
        if export_artifacts:
            with (self.run_dir / "artifacts" / "visualization_meta.json").open("w", encoding="utf-8") as f:
                json.dump(vis_meta, f, ensure_ascii=False, indent=2)

        correct = (preds == targets).astype(np.float64)
        conf = np.max(probs, axis=1)
        car = compute_coverage_at_risk(correct_mask=correct, confidence=conf, risk_threshold=0.05)

        sorted_unc = np.sort(uncertainty)
        split = max(1, int(0.1 * sorted_unc.size))
        in_scores = 1.0 - sorted_unc[:split]
        ood_scores = sorted_unc[-split:]
        fpr95 = compute_fpr95(in_scores=in_scores, ood_scores=ood_scores)

        cluster_idx = self._subsample_indices(high.shape[0], int(self.cfg.vsf.cluster_max_points), seed_offset=211)
        idx_high = compute_cluster_indices(high[cluster_idx], preds[cluster_idx])
        idx_low = compute_cluster_indices(low[cluster_idx], preds[cluster_idx])
        builder_warnings = getattr(self.granular_builder, "last_warning_counts", {})
        collision_pairs = getattr(self.decider, "last_collision_pairs", [])
        support_count = getattr(self.decider, "last_support_count", None)
        if isinstance(support_count, np.ndarray) and support_count.shape[0] == high.shape[0]:
            support_count_np = support_count.astype(np.int64, copy=False)
        else:
            if len(balls) == 0:
                support_count_np = np.zeros((high.shape[0],), dtype=np.int64)
            else:
                centers = np.stack([np.asarray(b.center, dtype=np.float64) for b in balls], axis=0)
                radii = np.asarray([max(float(b.radius), 0.0) for b in balls], dtype=np.float64)
                support_count_np = self._chunked_support_count(high, centers, radii)

        from tegb.types import REGION_BOUNDARY

        v3_boundary_rate = 0.0
        v3_multi_cover_rate = 0.0
        v3_collision_pair_count = 0.0
        v3_uncovered_rate = 0.0
        if self.cfg.decision.mode == "v3_spherical_collision" and self.cfg.loss.enable_3wd:
            v3_boundary_rate = float(np.mean(regions == REGION_BOUNDARY)) if regions.size else 0.0
            v3_multi_cover_rate = float(np.mean(support_count_np >= 2)) if support_count_np.size else 0.0
            v3_collision_pair_count = float(len(collision_pairs))
            v3_uncovered_rate = float(np.mean(support_count_np == 0)) if support_count_np.size else 0.0

        manifold_diag = compute_manifold_diagnostics(
            features=high,
            balls=balls,
            labels=targets,
            sample_cap=int(self.cfg.vsf.diagnostics_max_points),
            knn_k=int(self.cfg.vsf.diagnostics_knn_k),
            min_ball_members=int(self.cfg.vsf.diagnostics_min_ball_members),
            seed=int(self.cfg.experiment.seed + 313),
        )
        _progress_note("[Eval] compute manifold diagnostics")
        if export_artifacts:
            with (self.run_dir / "artifacts" / "manifold_diagnostics.json").open("w", encoding="utf-8") as f:
                json.dump(manifold_diag, f, ensure_ascii=False, indent=2)
        diag_summary = manifold_diag.get("summary", {})

        extras = {
            "warning_cov_fallback": float(builder_warnings.get("cov_fallback", 0)),
            "warning_chi2_fallback": float(builder_warnings.get("chi2_fallback", 0)),
            "warning_split_fallback": float(builder_warnings.get("split_fallback", 0)),
            "warning_split_non_converged": float(builder_warnings.get("split_non_converged", 0)),
            "warning_ph_fallback": float(builder_warnings.get("ph_fallback", 0)),
            "warning_empty_child": float(builder_warnings.get("empty_child", 0)),
            "warning_depth_limit": float(builder_warnings.get("depth_limit", 0)),
            "warning_low_entropy_gain": float(builder_warnings.get("low_entropy_gain", 0)),
            "collision_pairs": float(len(collision_pairs)),
            "v3_boundary_rate": float(v3_boundary_rate),
            "v3_multi_cover_rate": float(v3_multi_cover_rate),
            "v3_collision_pair_count": float(v3_collision_pair_count),
            "v3_uncovered_rate": float(v3_uncovered_rate),
            "visualization_files_count": float(len(vis_meta.get("files", []))),
            "visualization_warning_count": float(len(vis_meta.get("warnings", []))),
            "builder_exception": float(builder_exception),
            "train_split_count": float(len(getattr(self.train_loader, "dataset", []))),
            "val_split_count": float(len(getattr(self.eval_loader, "dataset", []))),
            "val_split_ratio": float(self.cfg.data.val_split_ratio),
            "eval_points_total": float(high.shape[0]),
            "vsf_eval_points": float(high_vsf.shape[0]),
            "cluster_eval_points": float(cluster_idx.shape[0]),
            "artifact_export_enabled": (1.0 if export_artifacts else 0.0),
            "cuda_alloc_before_eval_mb": float(cuda_alloc_before),
            "cuda_reserved_before_eval_mb": float(cuda_reserved_before),
            "diag_anisotropy_global_ratio": float(diag_summary.get("anisotropy_global_ratio", 1.0)),
            "diag_anisotropy_ball_mean": float(diag_summary.get("anisotropy_ball_mean", 1.0)),
            "diag_anisotropy_ball_p90": float(diag_summary.get("anisotropy_ball_p90", 1.0)),
            "diag_sphericity_ball_mean": float(diag_summary.get("sphericity_ball_mean", 0.0)),
            "diag_knn_cross_ball_ratio": float(diag_summary.get("knn_cross_ball_ratio", 0.0)),
            "diag_knn_same_label_cross_ball_ratio": float(diag_summary.get("knn_same_label_cross_ball_ratio", 0.0)),
            "diag_knn_diff_label_same_ball_ratio": float(diag_summary.get("knn_diff_label_same_ball_ratio", 0.0)),
            "diag_assignment_uncovered_count": float(diag_summary.get("assignment_uncovered_count", 0)),
            "diag_assignment_overlap_conflicts": float(diag_summary.get("assignment_overlap_conflicts", 0)),
        }

        result = {
            "vsf_auc": vsf.vsf_auc,
            "gtf_auc": vsf.gtf_auc,
            "snh_auc": vsf.snh_auc,
            "fpr95": -1.0 if fpr95 is None else float(fpr95),
            "coverage_at_risk": -1.0 if car is None else float(car),
            "silhouette_high": float(idx_high.get("silhouette", 0.0)),
            "silhouette_2d": float(idx_low.get("silhouette", 0.0)),
            "calinski_high": float(idx_high.get("calinski_harabasz", 0.0)),
            "calinski_2d": float(idx_low.get("calinski_harabasz", 0.0)),
            "davies_high": float(idx_high.get("davies_bouldin", 0.0)),
            "davies_2d": float(idx_low.get("davies_bouldin", 0.0)),
            "warning_cov_fallback": extras["warning_cov_fallback"],
            "warning_chi2_fallback": extras["warning_chi2_fallback"],
            "warning_split_fallback": extras["warning_split_fallback"],
            "warning_split_non_converged": extras["warning_split_non_converged"],
            "warning_ph_fallback": extras["warning_ph_fallback"],
            "warning_empty_child": extras["warning_empty_child"],
            "warning_depth_limit": extras["warning_depth_limit"],
            "warning_low_entropy_gain": extras["warning_low_entropy_gain"],
            "builder_exception": extras["builder_exception"],
            "collision_pairs": extras["collision_pairs"],
            "v3_boundary_rate": extras["v3_boundary_rate"],
            "v3_multi_cover_rate": extras["v3_multi_cover_rate"],
            "v3_collision_pair_count": extras["v3_collision_pair_count"],
            "v3_uncovered_rate": extras["v3_uncovered_rate"],
            "visualization_files_count": extras["visualization_files_count"],
            "visualization_warning_count": extras["visualization_warning_count"],
            "train_split_count": extras["train_split_count"],
            "val_split_count": extras["val_split_count"],
            "val_split_ratio": extras["val_split_ratio"],
            "eval_points_total": extras["eval_points_total"],
            "vsf_eval_points": extras["vsf_eval_points"],
            "cluster_eval_points": extras["cluster_eval_points"],
            "artifact_export_enabled": extras["artifact_export_enabled"],
            "cuda_alloc_before_eval_mb": extras["cuda_alloc_before_eval_mb"],
            "cuda_reserved_before_eval_mb": extras["cuda_reserved_before_eval_mb"],
            "diag_anisotropy_global_ratio": extras["diag_anisotropy_global_ratio"],
            "diag_anisotropy_ball_mean": extras["diag_anisotropy_ball_mean"],
            "diag_anisotropy_ball_p90": extras["diag_anisotropy_ball_p90"],
            "diag_sphericity_ball_mean": extras["diag_sphericity_ball_mean"],
            "diag_knn_cross_ball_ratio": extras["diag_knn_cross_ball_ratio"],
            "diag_knn_same_label_cross_ball_ratio": extras["diag_knn_same_label_cross_ball_ratio"],
            "diag_knn_diff_label_same_ball_ratio": extras["diag_knn_diff_label_same_ball_ratio"],
            "diag_assignment_uncovered_count": extras["diag_assignment_uncovered_count"],
            "diag_assignment_overlap_conflicts": extras["diag_assignment_overlap_conflicts"],
        }
        cache_dir = arrays.get("cache_dir")
        del high, low, alpha_all, probs, targets, preds, uncertainty, out, arrays
        del balls, support_count_np, vis_meta, vsf, idx_high, idx_low, high_vsf, low_vsf, probs_vsf
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cache_dir is not None:
            shutil.rmtree(cache_dir, ignore_errors=True)
        cuda_alloc_after, cuda_reserved_after = self._cuda_memory_mb()
        extras["cuda_alloc_after_eval_mb"] = float(cuda_alloc_after)
        extras["cuda_reserved_after_eval_mb"] = float(cuda_reserved_after)
        self.last_eval_extras = extras
        result["cuda_alloc_after_eval_mb"] = extras["cuda_alloc_after_eval_mb"]
        result["cuda_reserved_after_eval_mb"] = extras["cuda_reserved_after_eval_mb"]
        _progress_note("[Eval] done")
        return result
