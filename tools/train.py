#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/train.py  — STABLE PRODUCTION VERSION FOR 201-SEQUENCE DATASET
=====================================================================

DESIGN PRINCIPLES:
==================
1. NO MID-TRAINING UNFREEZING: Backbone layers remain static throughout training.
   Only layer4+layer5 are trainable from epoch 0 to end.

2. BATCHNORM ALWAYS FROZEN: BN layers kept in eval() mode to avoid noisy
   running statistics on small batch sizes (BS=4).

3. CONSERVATIVE LOC_WEIGHT RAMP: Starts at 0.05, reaches target 0.3 only
   at epoch 34 (85% of 40 epochs). Prevents regression head from dominating
   before classification head stabilizes.

4. SIMPLE COSINE LR SCHEDULE: 3-epoch linear warmup → cosine decay to 10% peak.
   No plateau detection complexity that can cause optimizer desync.

5. STRICT GRADIENT CLIPPING: Global clip at 1.0 prevents any single batch
   from corrupting weights on small datasets.

6. REDUCED OVERFITTING GUARDS: Lower VIDEOS_PER_EPOCH + stronger augmentation
   in config prevents memorization of 201 sequences.

All unstable mechanisms removed:
  ✗ No dynamic backbone unfreezing
  ✗ No BN unfreezing
  ✗ No gradient-spike recovery (strict clipping handles this)
  ✗ No complex plateau LR halving
  ✗ No optimizer rebuild mid-training
"""

import os
import sys
import time
import datetime
import signal
import logging
import argparse
import json
import yaml
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

DEFAULT_CFG = str(_PROJ_ROOT / 'configs' / 'hiFT_finetune.yaml')


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else (_PROJ_ROOT / p).resolve()


from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.utils.log_helper import init_log, add_file_handler
from pysot.utils.distributed import dist_init

# ── Constants — STABLE VALUES FOR SMALL DATASETS ─────────────────────────────
ACCUM_STEPS            = 4      # gradient accumulation (effective BS=16)
GRAD_NORM_WINDOW       = 10     # rolling window for monitoring only
GNORM_SPIKE_RATIO      = 3.0   # for logging spikes (not skipping steps)
LOC_WEIGHT_RAMP_FRAC   = 0.85  # LOC_WEIGHT reaches target at 85% of training
GRAD_CLIP_GLOBAL       = 1.0   # strict global clipping prevents corruption


def get_logger(name='global'):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            '[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(h)
    return logger


logger = get_logger('global')


# ── Config validation ─────────────────────────────────────────────────────────

def assert_config_consistency():
    errors = []
    s = cfg.TRAIN.OUTPUT_SIZE
    f = cfg.TRAIN.OUTPUTFEATURE_SIZE
    if s != f:
        errors.append(f"TRAIN.OUTPUT_SIZE ({s}) != TRAIN.OUTPUTFEATURE_SIZE ({f}).")
    if s != 11:
        errors.append(
            f"TRAIN.OUTPUT_SIZE={s} but expected 11 for "
            f"SEARCH_SIZE={cfg.TRAIN.SEARCH_SIZE}.")
    if errors:
        for e in errors:
            logger.error(f"[CONFIG] {e}")
        raise RuntimeError("Config consistency check failed.")
    logger.info(f"[CONFIG] OUTPUT_SIZE={s} consistent ✓")


# ── LOC_WEIGHT SCHEDULE — SLOW RAMP FOR STABILITY ─────────────────────────────

def get_loc_weight(epoch: int,
                   total_epochs: int,
                   target: float,
                   start: float = 0.05,
                   ramp_frac: float = 0.60) -> float:
    """
    Conservative LOC_WEIGHT schedule for small datasets.

    Design:
      - Linear ramp from `start` → `target` over first `ramp_frac` of training
      - Held at `target` for final portion

    Example (target=0.3, total=40, start=0.05, ramp_frac=0.85):
      epoch 0:  0.050  ← classification dominates
      epoch 4:  0.086  ← still cls-focused
      epoch 8:  0.122  ← gentle regression introduction
      epoch 16: 0.194  ← balanced
      epoch 34: 0.300  ← target reached
      epoch 39: 0.300  ← held
    """
    ramp_epochs = max(1, int(total_epochs * ramp_frac))
    if epoch >= ramp_epochs:
        return float(target)
    progress = epoch / ramp_epochs
    return start + (target - start) * progress


# ── Weight loading ────────────────────────────────────────────────────────────

def _strip(state):
    return {k.replace('module.', ''): v for k, v in state.items()}


def load_weights(model, optimizer, scheduler, args):
    if args.resume:
        path = Path(args.resume)
        if not path.exists():
            raise FileNotFoundError(f"--resume path not found: {path}")
        return _load_full_ckpt(model, optimizer, scheduler, path)
    if not args.no_resume:
        auto = _abs(cfg.TRAIN.SNAPSHOT_DIR) / 'latest.pth'
        if auto.exists():
            logger.info(f"[WEIGHTS] Auto-resuming: {auto}")
            return _load_full_ckpt(model, optimizer, scheduler, auto)
    candidates = [
        _abs(cfg.BACKBONE.PRETRAINED),
        _PROJ_ROOT / 'pretrained' / 'first.pth',
        _PROJ_ROOT / 'pretrained_models' / 'first.pth',
    ]
    for p in candidates:
        if Path(p).exists():
            _load_pretrained(model, Path(p))
            return 0
    logger.warning("[WEIGHTS] No pretrained weights found. Training from scratch.")
    return 0


def _load_full_ckpt(model, optimizer, scheduler, path):
    ckpt  = torch.load(str(path), map_location='cpu')
    state = _strip(ckpt.get('state_dict', ckpt))
    miss, unexp = model.load_state_dict(state, strict=False)
    miss = [k for k in miss if 'num_batches_tracked' not in k]
    logger.info(f"[WEIGHTS] Loaded {path.name}: missing={len(miss)} "
                f"unexpected={len(unexp)}")
    if 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            for st in optimizer.state.values():
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        st[k] = v.cuda()
        except Exception as e:
            logger.warning(f"[WEIGHTS] Optimizer restore failed: {e}")
    if 'scheduler' in ckpt and scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt['scheduler'])
        except Exception as e:
            logger.warning(f"[WEIGHTS] Scheduler restore failed: {e}")
    ep = ckpt.get('epoch', -1) + 1
    logger.info(f"[WEIGHTS] Resuming from epoch {ep}")
    return ep


def _load_pretrained(model, path: Path):
    logger.info(f"[WEIGHTS] Loading pretrained: {path}")
    raw = torch.load(str(path), map_location='cpu')
    if isinstance(raw, dict):
        state = raw.get('state_dict', raw.get('model', raw))
    else:
        state = raw
    state = _strip(state)
    has_backbone = any(k.startswith('backbone.') for k in state)
    has_grader   = any(k.startswith('grader.')   for k in state)
    if has_backbone and has_grader:
        miss, _ = model.load_state_dict(state, strict=False)
        miss = [k for k in miss if 'num_batches_tracked' not in k]
        logger.info(f"[WEIGHTS] Full HiFT: missing={len(miss)}")
    elif has_backbone:
        bb_state = {k[len('backbone.'):]: v for k, v in state.items()
                    if k.startswith('backbone.')}
        miss, _ = model.backbone.load_state_dict(bb_state, strict=False)
        miss = [k for k in miss if 'num_batches_tracked' not in k]
        logger.warning(f"[WEIGHTS] Backbone-only: missing={len(miss)}")
    else:
        miss, _ = model.load_state_dict(state, strict=False)
        miss_f = [k for k in miss if 'num_batches_tracked' not in k]
        if len(miss_f) > 20:
            bb_state = {k.replace('features.', ''): v
                        for k, v in state.items()}
            miss2, _ = model.backbone.load_state_dict(bb_state, strict=False)
            logger.info(f"[WEIGHTS] Bare backbone: missing={len(miss2)}")
        else:
            logger.info(f"[WEIGHTS] Bare keys: missing={len(miss_f)}")


# ── Backbone management — STATIC FREEZING ─────────────────────────────────────

def freeze_bn(model):
    """Freeze ALL BatchNorm layers permanently in eval mode."""
    count = 0
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            count += 1
    return count


def set_backbone_trainable(model, trainable_layers):
    """
    Static backbone freezing — safe version
    """
    frozen = 0
    trainable = 0

    for name, param in model.backbone.named_parameters():
        param.requires_grad = False

        for layer in trainable_layers:
            if layer in name:
                param.requires_grad = True
                break

        if param.requires_grad:
            trainable += 1
        else:
            frozen += 1

    logger.info(f"Backbone: {frozen} frozen, {trainable} trainable "
                f"(training: {trainable_layers})")


# ── Optimizer / Scheduler — SIMPLE & STABLE ───────────────────────────────────

def build_optimizer(model, base_lr: float):
    """
    AdamW with layer-specific LRs.
    Backbone: base_lr * 0.1 (conservative fine-tuning)
    Head:     base_lr (full learning rate)
    """
    backbone_trainable = [p for p in model.backbone.parameters()
                          if p.requires_grad]
    head_params = list(model.grader.parameters())

    param_groups = []
    if backbone_trainable:
        param_groups.append({
            'params': backbone_trainable,
            'lr': base_lr * 0.1,
            'weight_decay': 5e-4,
            'name': 'backbone'
        })
    param_groups.append({
        'params': head_params,
        'lr': base_lr,
        'weight_decay': 5e-4,
        'name': 'head'
    })

    return torch.optim.AdamW(param_groups)


def build_cosine_scheduler(optimizer, total_epochs: int,
                            warmup_epochs: int = 3,
                            min_lr_frac: float = 0.1):
    """
    Simple, stable LR schedule:
      - Linear warmup for `warmup_epochs` (0.01 → 1.0)
      - Cosine decay from peak to `min_lr_frac * peak`
    No plateau detection, no hold phases — simplicity = stability.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 1% to 100% of peak LR
            return 0.01 + 0.99 * (epoch + 1) / max(warmup_epochs, 1)
        # Cosine decay
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Validation ────────────────────────────────────────────────────────────────
import cv2


def _parse_init_bbox(anno_path):
    with open(str(anno_path), 'r', newline='') as f:
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    x, y, w, h = map(float, parts[:4])
                    if w > 0 and h > 0:
                        return [x, y, w, h]
                except ValueError:
                    continue
    return None


def _load_gt(anno_path):
    bboxes = []
    with open(str(anno_path), 'r', newline='') as f:
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    x, y, w, h = map(float, parts[:4])
                    bboxes.append([x, y, w, h])
                except ValueError:
                    bboxes.append([0, 0, 0, 0])
    return bboxes


def _iou_xywh(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-6)


def _center_err(b1, b2):
    cx1, cy1 = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    cx2, cy2 = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def run_validation(model, manifest_path, base_dir, max_seqs=50):
    """
    Evaluate on internal validation split.
    Returns dict with metrics or None if validation cannot run.
    """
    from pysot.tracker.hift_tracker import HiFTTracker

    if not manifest_path or not Path(manifest_path).exists():
        return None

    contest_manifest_path = base_dir / 'metadata' / 'contestant_manifest.json'
    if not contest_manifest_path.exists():
        return None

    with open(str(contest_manifest_path), 'r', encoding='utf-8') as f:
        contest_manifest = json.load(f)
    all_seqs = {**contest_manifest.get('train', {}),
                **contest_manifest.get('public_lb', {})}

    with open(str(manifest_path), 'r', encoding='utf-8') as f:
        val_annos = json.load(f)

    # ===== SAFE sequence length extraction =====
    seq_lengths = {}
    for k, v in val_annos.items():
        try:
            obj = next(iter(v.values()))
            frames = obj.get('frames', [])
            seq_lengths[k] = len(frames)
        except Exception:
            seq_lengths[k] = 0

    # ===== Stratified selection (NO SHUFFLE) =====
    sorted_seqs = sorted(seq_lengths.items(), key=lambda x: x[1])

    if len(sorted_seqs) == 0:
        return None

    step = max(1, len(sorted_seqs) // max_seqs)

    seq_keys = []
    for i in range(0, len(sorted_seqs), step):
        seq_keys.append(sorted_seqs[i][0])
        if len(seq_keys) >= max_seqs:
            break

    model.eval()
    all_ious, all_cerrs = [], []

    for seq_key in seq_keys:
        if seq_key not in all_seqs:
            continue

        seq_info   = all_seqs[seq_key]
        video_path = base_dir / seq_info['video_path']
        anno_path  = base_dir / seq_info['annotation_path']

        if not video_path.exists() or not anno_path.exists():
            continue

        gt = _load_gt(anno_path)
        if len(gt) < 2:
            continue

        init_bbox = _parse_init_bbox(anno_path)
        if init_bbox is None:
            continue

        try:
            tracker = HiFTTracker(model)
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                continue

            ret, frame = cap.read()
            if not ret:
                cap.release()
                continue

            tracker.init(frame, init_bbox)

            all_ious.append(1.0)
            all_cerrs.append(0.0)

            last_pred = init_bbox

            for frame_idx in range(1, len(gt)):
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    out  = tracker.track(frame)
                    pred = out['bbox']
                    last_pred = pred
                except Exception:
                    pred = last_pred

                all_ious.append(_iou_xywh(pred, gt[frame_idx]))
                all_cerrs.append(_center_err(pred, gt[frame_idx]))

            cap.release()

        except Exception:
            continue

    model.train()

    if not all_ious:
        return None

    ious  = np.array(all_ious)
    cerrs = np.array(all_cerrs)

    thresholds = np.linspace(0, 1, 21)
    auc  = float(np.mean([np.mean(ious >= t) for t in thresholds]))
    prec = float(np.mean(cerrs < 20))

    return {
        'auc': auc,
        'precision': prec,
        'mean_iou': float(np.mean(ious)),
        'n_frames': len(ious)
    }


# ── SafeTrainer — ATOMIC CHECKPOINTS ──────────────────────────────────────────

class SafeTrainer:
    def __init__(self, model, optimizer, scheduler, save_dir):
        self.model       = model
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.save_dir    = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interrupted = False
        signal.signal(signal.SIGINT,  self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        logger.warning("Interrupt received — will save checkpoint after this epoch.")
        self.interrupted = True

    def save(self, epoch, filename, extra=None):
        state = {
            'epoch':      epoch,
            'state_dict': self.model.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'scheduler':  self.scheduler.state_dict()
                          if self.scheduler else None,
        }
        if extra:
            state.update(extra)
        path = self.save_dir / filename
        tmp  = self.save_dir / (filename + '.tmp')
        torch.save(state, str(tmp))
        tmp.replace(path)  # atomic write
        logger.info(f"  → Saved: {path.name}")
        return path

    def restore(self, filename):
        """Restore model + optimizer from checkpoint."""
        path = self.save_dir / filename
        if not path.exists():
            logger.warning(f"[RESTORE] {path} not found, skipping.")
            return False
        ckpt  = torch.load(str(path), map_location='cpu')
        state = {k.replace('module.', ''): v
                 for k, v in ckpt.get('state_dict', ckpt).items()}
        self.model.load_state_dict(state, strict=False)
        if 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                for st in self.optimizer.state.values():
                    for k, v in st.items():
                        if isinstance(v, torch.Tensor):
                            st[k] = v.cuda()
            except Exception:
                pass
        logger.info(f"[RESTORE] Restored model from {path.name}")
        return True


# ── EarlyStopper — SIMPLE PATIENCE ────────────────────────────────────────────

class EarlyStopper:
    """
    Simple early stopper based on validation AUC.
    Activates only after warmup period.
    """
    def __init__(self, patience: int = 8, delta: float = 1e-4, warmup: int = 5):
        self.patience  = patience
        self.delta     = delta
        self.warmup    = warmup
        self.best_auc  = -1.0
        self.no_improve = 0

    def step(self, epoch, val_auc):
        if epoch < self.warmup or val_auc is None:
            return False
        if val_auc > self.best_auc + self.delta:
            self.best_auc = val_auc
            self.no_improve = 0
            return False
        self.no_improve += 1
        if self.no_improve >= self.patience:
            logger.info(f"[EARLY-STOP] No val improvement for {self.patience} epochs. "
                        f"Best AUC: {self.best_auc:.4f}")
            return True
        return False


# ── Training loop — STRICT CLIPPING, NO SPIKE SKIPPING ────────────────────────

def train_one_epoch(model, dataloader, optimizer, loc_w, cls_w, grad_clip):
    """
    Single-epoch training with:
      - Gradient accumulation
      - Strict global gradient clipping (no spike skipping)
      - Simple, deterministic updates

    Returns: avg_loss, avg_cls, avg_loc, avg_gnorm
    """
    model.train()
    step_losses, step_cls, step_loc, step_gnorms = [], [], [], []
    optimizer.zero_grad(set_to_none=True)
    pending = 0

    for i, data in enumerate(dataloader):
        try:
            outputs = model(data, loc_weight=loc_w, cls_weight=cls_w)
        except RuntimeError as e:
            if 'decode_loc' in str(e) or 'OUTPUT_SIZE' in str(e):
                raise
            logger.warning(f"[TRAIN] Batch {i} forward error: {e}")
            continue

        loss = outputs['total_loss']
        if not torch.isfinite(loss):
            logger.warning(f"[TRAIN] Non-finite loss at batch {i}: {loss.item():.4f}")
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            continue

        (loss / ACCUM_STEPS).backward()
        pending += 1

        step_losses.append(outputs['total_loss'].item())
        step_cls.append(float(outputs['cls_loss']))
        step_loc.append(float(outputs['loc_loss']))

        if pending == ACCUM_STEPS:
            # Strict global clipping BEFORE step
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip))
            step_gnorms.append(gnorm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

    # Flush remaining gradients
    if pending > 0:
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip))
        step_gnorms.append(gnorm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss  = float(np.mean(step_losses)) if step_losses else float('inf')
    avg_cls   = float(np.mean(step_cls))   if step_cls   else 0.0
    avg_loc   = float(np.mean(step_loc))   if step_loc   else 0.0
    avg_gnorm = float(np.mean(step_gnorms)) if step_gnorms else 0.0
    return avg_loss, avg_cls, avg_loc, avg_gnorm


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='HiFT fine-tuning — STABLE PRODUCTION VERSION')
    p.add_argument('--cfg',          default=DEFAULT_CFG,
                   help='Path to YAML config file')
    p.add_argument('--resume',       default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--no-resume',    action='store_true',
                   help='Do not auto-resume from latest.pth')
    p.add_argument('--val-manifest', type=str, default=None,
                   help='Path to val.json crop annotation file')
    p.add_argument('--val-seqs',     type=int, default=20,
                   help='Number of sequences for validation')
    # These are now FIXED in code/config for stability
    return p.parse_args()


def _resolve_val_manifest(args):
    if args.val_manifest:
        path = Path(args.val_manifest)
        if not path.exists():
            raise FileNotFoundError(f"--val-manifest not found: {path}")
        return path.resolve()
    for candidate_name in ['val.json', 'manifest_val.json']:
        default = _PROJ_ROOT / 'data' / 'processed' / candidate_name
        if default.exists():
            logger.info(f"[VAL-MANIFEST] Using: {default}")
            return default
    logger.warning("[VAL-MANIFEST] Not found — using train-loss early stopping only.")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    from yacs.config import CfgNode as CN
    cfg.merge_from_other_cfg(CN(cfg_dict))

    assert_config_consistency()

    os.makedirs(_abs(cfg.TRAIN.LOG_DIR), exist_ok=True)
    try:
        add_file_handler('global',
                         str(_abs(cfg.TRAIN.LOG_DIR) / 'train.log'))
    except Exception:
        pass

    logger.info(f"Project root : {_PROJ_ROOT}")
    logger.info(f"Config       : {args.cfg}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | "
                    f"VRAM: {props.total_memory / 1024**3:.1f}GB")

    dist_init()

    contest_dir = Path(os.environ.get(
        'CONTEST_DATA_DIR',
        str(_PROJ_ROOT.parent / 'AIC4-UAV-Tracker' /
            'data' / 'contest_release')
    )).resolve()

    val_manifest_path = _resolve_val_manifest(args)

    # ── Build model ───────────────────────────────────────────────────────────
    logger.info("Building model...")
    model = ModelBuilder()
    model = model.cuda()

    total_epochs = cfg.TRAIN.EPOCH
    base_lr      = cfg.TRAIN.BASE_LR
    warmup_ep    = cfg.TRAIN.LR_WARMUP.EPOCH  # 3

    # STATIC BACKBONE FREEZING: Only layer4+layer5 trainable, FOREVER
    trainable_layers = ['layer4', 'layer5']
    set_backbone_trainable(model, trainable_layers)

    # PERMANENTLY FREEZE BATCHNORM
    n_frozen_bn = freeze_bn(model)
    logger.info(f"[BN] Frozen {n_frozen_bn} BN layers (eval mode, no training)")

    optimizer = build_optimizer(model, base_lr)
    lr_cfg = cfg.TRAIN.LR
    if lr_cfg.TYPE == 'cosine':
        scheduler = build_cosine_scheduler(optimizer, total_epochs, 
                                        warmup_epochs=cfg.TRAIN.LR_WARMUP.EPOCH,
                                        min_lr_frac=lr_cfg.KWARGS.end_lr / lr_cfg.KWARGS.start_lr)
    elif lr_cfg.TYPE == 'log':
        # Implement log decay or use YACS built-in
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unknown LR type: {lr_cfg.TYPE}")

    start_epoch = load_weights(model, optimizer, scheduler, args)

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    dataset    = TrkDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.TRAIN.NUM_WORKERS > 0)
    )
    logger.info(f"Dataset: {len(dataset)} samples | "
                f"Batch: {cfg.TRAIN.BATCH_SIZE} | "
                f"Effective (×{ACCUM_STEPS}): "
                f"{cfg.TRAIN.BATCH_SIZE * ACCUM_STEPS}")

    # ── Training support objects ──────────────────────────────────────────────
    trainer = SafeTrainer(model, optimizer, scheduler,
                          _abs(cfg.TRAIN.SNAPSHOT_DIR))
    stopper = EarlyStopper(
        patience = 8,
        delta    = 1e-4,
        warmup   = max(5, warmup_ep + 2)
    )

    best_auc    = 0.0
    best_loss   = float('inf')
    overall_t   = time.time()

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_csv = _abs(cfg.TRAIN.LOG_DIR) / 'training_log.csv'
    if not log_csv.exists():
        with open(log_csv, 'w') as f:
            f.write('epoch,train_loss,cls_loss,loc_loss,lr_head,'
                    'val_auc,val_precision,val_iou,'
                    'grad_norm,loc_weight\n')

    logger.info(
        f"\n{'='*70}\n"
        f"STABLE PRODUCTION TRAINING (201-sequence dataset)\n"
        f"  Epochs:          {start_epoch+1} → {total_epochs}\n"
        f"  LOC_WEIGHT:      0.05 → {cfg.TRAIN.LOC_WEIGHT:.3f} (ramp to 85%)\n"
        f"  Base LR:         {base_lr:.2e}\n"
        f"  Grad clip:       {GRAD_CLIP_GLOBAL}\n"
        f"  Backbone:        layer4+layer5 ONLY (static freeze)\n"
        f"  BatchNorm:       FROZEN (eval mode)\n"
        f"  VIDEOS/epoch:    {cfg.DATASET.VIDEOS_PER_EPOCH}\n"
        f"  Batch (eff):     {cfg.TRAIN.BATCH_SIZE * ACCUM_STEPS}\n"
        f"{'='*70}"
    )

    for epoch in range(start_epoch, total_epochs):
        if trainer.interrupted:
            break

        # ── Epoch LOC_WEIGHT (slow ramp) ──────────────────────────────────────
        loc_w = get_loc_weight(
            epoch,
            total_epochs,
            cfg.TRAIN.LOC_WEIGHT,
            start=0.05,
            ramp_frac=LOC_WEIGHT_RAMP_FRAC
        )
        cls_w  = cfg.TRAIN.CLS_WEIGHT
        cur_lr = optimizer.param_groups[-1]['lr']  # head LR

        logger.info(
            f"\n{'='*65}\n"
            f"Epoch [{epoch+1}/{total_epochs}] | "
            f"LOC_W: {loc_w:.4f} | "
            f"LR(head): {cur_lr:.2e} | "
            f"Backbone: {trainable_layers}"
        )

        # ── Shuffle dataset ───────────────────────────────────────────────────
        dataset.pick = dataset.shuffle()

        # ── Train epoch ───────────────────────────────────────────────────────
        epoch_t = time.time()
        avg_loss, avg_cls, avg_loc, avg_gnorm = train_one_epoch(
            model, dataloader, optimizer, loc_w, cls_w,
            grad_clip = GRAD_CLIP_GLOBAL,
        )
        scheduler.step()

        epoch_time = time.time() - epoch_t
        logger.info(
            f"Epoch {epoch+1} | {epoch_time:.0f}s | "
            f"loss={avg_loss:.4f} (cls={avg_cls:.4f} loc={avg_loc:.4f}) | "
            f"gnorm={avg_gnorm:.2f} | lr={cur_lr:.2e}"
        )

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = None
        if args.val_seqs > 0 and val_manifest_path:
            val_metrics = run_validation(
                model, val_manifest_path, contest_dir,
                max_seqs=args.val_seqs)
            if val_metrics:
                logger.info(
                    f"[VAL] AUC={val_metrics['auc']:.4f} | "
                    f"Prec={val_metrics['precision']:.4f} | "
                    f"mIoU={val_metrics['mean_iou']:.4f} | "
                    f"frames={val_metrics['n_frames']}")

        # ── Checkpoint saving ─────────────────────────────────────────────────
        extra = {
            'train_loss': avg_loss,
            'val_auc':    val_metrics['auc'] if val_metrics else None,
            'loc_weight': loc_w,
        }
        trainer.save(epoch, 'latest.pth', extra)

        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save(epoch, 'best_train.pth', extra)
            logger.info(f"  ★ New best train loss: {best_loss:.4f}")

        if val_metrics and val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            trainer.save(epoch, 'best_val.pth', extra)
            trainer.save(epoch, 'best.pth', extra)  # for submit.py
            logger.info(f"  ★ New best val AUC: {best_auc:.4f}")

        # ── CSV logging ───────────────────────────────────────────────────────
        with open(log_csv, 'a') as f:
            f.write(
                f"{epoch+1},{avg_loss:.6f},{avg_cls:.6f},{avg_loc:.6f},"
                f"{cur_lr:.8f},"
                f"{val_metrics['auc']       if val_metrics else ''},"
                f"{val_metrics['precision'] if val_metrics else ''},"
                f"{val_metrics['mean_iou']  if val_metrics else ''},"
                f"{avg_gnorm:.4f},{loc_w:.4f}\n"
            )

        torch.cuda.empty_cache()

        # ── Early stopping check ──────────────────────────────────────────────
        val_auc_for_stop = val_metrics['auc'] if val_metrics else None
        if stopper.step(epoch, val_auc=val_auc_for_stop):
            logger.info(
                f"[EARLY-STOP] Triggered at epoch {epoch+1}. "
                f"Best val AUC: {best_auc:.4f}")
            break

        if trainer.interrupted:
            trainer.save(epoch, 'interrupted.pth', extra)
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    total_t = time.time() - overall_t
    logger.info(
        f"\n{'='*65}\n"
        f"Training complete!\n"
        f"  Duration:       {str(datetime.timedelta(seconds=int(total_t)))}\n"
        f"  Best val AUC:   {best_auc:.4f}\n"
        f"  Best train loss:{best_loss:.4f}\n"
        f"  Checkpoints:    {_abs(cfg.TRAIN.SNAPSHOT_DIR)}\n"
        f"{'='*65}"
    )


if __name__ == '__main__':
    main()