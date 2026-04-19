#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/train.py — CORRECTED PRODUCTION VERSION v2.1
=================================================

FIXES APPLIED IN THIS VERSION (on top of v2):

[CRITICAL] BUG-F: Double scheduler.step() call + wrong LR source.
  The original v2 code called scheduler.step() BEFORE training, then again
  AFTER training, causing LR to decay 2× faster than intended. Additionally,
  it manually recomputed scheduler_mult instead of using the scheduler's
  actual output. Fix: Call scheduler.step() ONCE after training, then apply
  backbone shielding on top of the scheduler's computed LR values.

All earlier fixes from v2 are retained.
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

# ── Constants ─────────────────────────────────────────────────────────────────
ACCUM_STEPS          = 4       # gradient accumulation → effective BS=16

# FIX BUG-B: raised from 1.0 to 5.0.
# Pre-clip gnorm in log was 4.0 at epoch 1, meaning the 1.0 clip was
# suppressing gradients by 4× on every step. 5.0 allows natural gradient flow
# while still preventing extreme spikes.
GRAD_CLIP_GLOBAL     = 5.0

# FIX BUG-A: raised from 3 to 5.
# The LR warmup runs for 3 epochs reaching peak at epoch 3.
# With freeze=3, loc_weight turned on at epoch 4 exactly at peak LR.
# Extending to 5 gives 2 epochs of cosine descent before loc_weight starts,
# reducing the disruption to the classification head's learned features.
LOC_WEIGHT_FREEZE_EP = 25      # <<< CHANGE: Freeze for 25 epochs instead of 10
LOC_WEIGHT_RAMP_FRAC = 0.50    # <<< CHANGE: Slower ramp-up speed
MAX_VAL_FRAMES       = 2000    # hard frame budget per validation run

# FIX BUG-C: backbone gradient multiplier during loc transition window.
# For BACKBONE_LOC_SHIELD_EPOCHS epochs after loc_weight turns on,
# the backbone LR is reduced to BACKBONE_SHIELD_LR_MULT of its normal value.
# This insulates the backbone features from noisy regression gradients
# while the loc head learns to produce meaningful outputs.
BACKBONE_LOC_SHIELD_EPOCHS = 3
BACKBONE_SHIELD_LR_MULT    = 0.1   # 10% of normal backbone LR during shield


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


# ── LOC_WEIGHT SCHEDULE ───────────────────────────────────────────────────────

def get_loc_weight(epoch: int, total_epochs: int, target: float,
                   start: float = 0.05,
                   freeze_epochs: int = LOC_WEIGHT_FREEZE_EP,
                   ramp_frac: float = LOC_WEIGHT_RAMP_FRAC) -> float:
    """
    Conservative LOC_WEIGHT schedule:

      Epochs 0..(freeze_epochs-1):   LOC_WEIGHT = 0.0
        Classification head stabilises; loc head (zero-initialized) produces
        near-zero outputs, so its IoU loss would be ~0 anyway.

      Epochs freeze_epochs..(freeze_epochs + ramp_epochs): linear ramp start→target
        loc_weight grows slowly from `start` to `target`.

      Remaining epochs: held at target.

    With freeze_epochs=5, total=50, ramp_frac=0.70, target=0.35:
      epoch 0-4:  0.000  (freeze — LR peaks at epoch 3, cools by epoch 5)
      epoch 5:    0.050  (ramp starts, LR already on cosine descent)
      epoch ~37:  0.350  (target, 70% of remaining 45 epochs)
      epoch 38-49: 0.350 (held)
    """
    if epoch < freeze_epochs:
        return 0.0
    active_epochs = total_epochs - freeze_epochs
    ramp_epochs   = max(1, int(active_epochs * ramp_frac))
    progress      = (epoch - freeze_epochs) / ramp_epochs
    progress      = min(progress, 1.0)
    return start + (target - start) * progress


# Find this function and replace it with:
def get_backbone_lr_mult(epoch: int, base_mult: float, is_backbone: bool = None) -> float:
    """
    FIX BUG-C: Additional backbone LR reduction during loc transition.
    For BACKBONE_LOC_SHIELD_EPOCHS epochs after LOC_WEIGHT_FREEZE_EP,
    the backbone LR multiplier is further reduced to insulate backbone
    features from noisy loc head gradients.
    Returns the multiplier to apply ON TOP of the normal backbone LR ratio.
    NOTE: Only applies shield to backbone (base_mult < 1.0), not head.
    """
    loc_start = LOC_WEIGHT_FREEZE_EP
    shield_end = loc_start + BACKBONE_LOC_SHIELD_EPOCHS
    # Use explicit flag if available, otherwise fallback to base_mult heuristic
    is_bb = is_backbone if is_backbone is not None else (base_mult < 1.0)
    if is_bb and loc_start <= epoch < shield_end:
        return base_mult * BACKBONE_SHIELD_LR_MULT
    return base_mult

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


# ── Backbone management ───────────────────────────────────────────────────────

def freeze_bn(model):
    """
    Freeze ALL BatchNorm layers in the backbone permanently.
    Returns list of frozen modules so re_freeze_bn() can restore them cheaply.
    """
    frozen_modules = []
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad   = False
            frozen_modules.append(m)
    logger.info(f"[BN] Frozen {len(frozen_modules)} BN layers (eval mode, no grad)")
    return frozen_modules


def re_freeze_bn(frozen_modules):
    """
    Restore frozen BN layers to eval mode after any model.train() call.

    model.train() recursively resets ALL submodules to training mode, which
    undoes the .eval() state set by freeze_bn(). This must be called after
    every model.train() call to restore the frozen BN state. Without this,
    BN running statistics update during training steps after each validation
    round, introducing increasingly noisy normalisation.
    """
    for m in frozen_modules:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad   = False


def set_backbone_trainable(model, trainable_layers):
    """Static backbone freezing — only named layers are trainable."""
    frozen   = 0
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


# ── Optimizer / Scheduler ─────────────────────────────────────────────────────

def build_optimizer(model, base_lr: float):
    backbone_trainable = [p for p in model.backbone.parameters()
                          if p.requires_grad]
    head_params = list(model.grader.parameters())
    param_groups = []
    if backbone_trainable:
        param_groups.append({
            'params':       backbone_trainable,
            'lr':           base_lr,
            'weight_decay': 5e-4,
            'name':         'backbone',
            'base_lr_mult': 0.1,
            'is_backbone':  True,   # ← ADD THIS LINE
        })
    param_groups.append({
        'params':       head_params,
        'lr':           base_lr,
        'weight_decay': 5e-4,
        'name':         'head',
        'base_lr_mult': 1.0,
        'is_backbone':  False,  # ← ADD THIS LINE
    })
    return torch.optim.AdamW(param_groups)


def apply_backbone_lr_shield(optimizer, epoch: int, scheduler_lr: float,
                              base_lr: float):
    """
    FIX BUG-C: Apply backbone LR shielding during the loc transition window.

    During BACKBONE_LOC_SHIELD_EPOCHS epochs after LOC_WEIGHT_FREEZE_EP,
    the backbone learning rate is further reduced to 10% of its normal value.
    This prevents the randomly-initialized loc head's gradients from corrupting
    the backbone features that were carefully learned during the cls-only phase.
    """
    for pg in optimizer.param_groups:
        base_mult = pg.get('base_lr_mult', 1.0)
        shielded_mult = get_backbone_lr_mult(epoch, base_mult)
        pg['lr'] = scheduler_lr * shielded_mult


def build_cosine_scheduler(optimizer, total_epochs: int,
                            warmup_epochs: int = 5,
                            min_lr_frac: float = 0.1):
    """
    LR schedule: linear warmup from 10%→100% of base_lr over warmup_epochs,
    then cosine decay from 100%→10% over remaining epochs.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
        progress = min(progress, 1.0)
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
    """None-safe IoU between two [x, y, w, h] boxes."""
    if b1 is None or b2 is None:
        return 0.0
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-6)


def _center_err(b1, b2):
    """None-safe center error. Returns large sentinel value on None."""
    if b1 is None or b2 is None:
        return 999.0
    cx1, cy1 = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    cx2, cy2 = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def run_validation(model, frozen_bn_modules, manifest_path, base_dir,
                   max_seqs=20, max_frames=MAX_VAL_FRAMES):
    """
    Evaluate on internal validation split with a hard frame budget.

    KEY FIXES retained from v1:
    - re_freeze_bn() called after model.train() to restore BN eval state.
    - None-safe bbox handling.
    - max_frames budget cap.
    - Init frame excluded from metrics (trivially IoU=1.0 by construction).
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

    # Stratified sequence selection by length
    seq_lengths = {}
    for k, v in val_annos.items():
        try:
            obj    = next(iter(v.values()))
            frames = obj.get('frames', [])
            seq_lengths[k] = len(frames)
        except Exception:
            seq_lengths[k] = 0

    sorted_seqs = sorted(seq_lengths.items(), key=lambda x: x[1])
    if not sorted_seqs:
        return None

    step      = max(1, len(sorted_seqs) // max_seqs)
    seq_keys  = [sorted_seqs[i][0]
                 for i in range(0, len(sorted_seqs), step)][:max_seqs]

    model.eval()

    all_ious  = []
    all_cerrs = []
    total_frames_used = 0

    for seq_key in seq_keys:
        if total_frames_used >= max_frames:
            break

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

        remaining       = max_frames - total_frames_used
        seq_frame_limit = min(len(gt), remaining, 300)

        try:
            tracker = HiFTTracker(model)
            cap     = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                continue

            ret, frame = cap.read()
            if not ret:
                cap.release()
                continue

            tracker.init(frame, init_bbox)
            last_pred = list(init_bbox)

            for frame_idx in range(1, seq_frame_limit):
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    out = tracker.track(frame)

                    if out.get('failed', False) or out['bbox'] is None:
                        pred = last_pred
                    else:
                        pred      = out['bbox']
                        last_pred = pred

                except Exception:
                    pred = last_pred

                all_ious.append(_iou_xywh(pred, gt[frame_idx]))
                all_cerrs.append(_center_err(pred, gt[frame_idx]))
                total_frames_used += 1

            cap.release()

        except Exception:
            continue

    # Restore training mode with BN frozen.
    model.train()
    re_freeze_bn(frozen_bn_modules)

    if not all_ious:
        return None

    ious  = np.array(all_ious)
    cerrs = np.array(all_cerrs)

    thresholds = np.linspace(0, 1, 21)
    auc  = float(np.mean([np.mean(ious >= t) for t in thresholds]))
    prec = float(np.mean(cerrs < 20))

    return {
        'auc':       auc,
        'precision': prec,
        'mean_iou':  float(np.mean(ious)),
        'n_frames':  len(ious)
    }


# ── SafeTrainer ───────────────────────────────────────────────────────────────

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
        logger.warning("Interrupt received — saving after this epoch.")
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
        tmp.replace(path)
        logger.info(f"  → Saved: {path.name}")
        return path


# ── EarlyStopper ──────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience: int = 6, delta: float = 1e-4, warmup: int = 5):
        self.patience   = patience
        self.delta      = delta
        # warmup matches LOC_WEIGHT_FREEZE_EP so we don't stop before loc kicks in
        self.warmup     = warmup
        self.best_auc   = -1.0
        self.no_improve = 0

    def step(self, epoch, val_auc):
        if epoch < self.warmup or val_auc is None:
            return False
        if val_auc > self.best_auc + self.delta:
            self.best_auc   = val_auc
            self.no_improve = 0
            return False
        self.no_improve += 1
        if self.no_improve >= self.patience:
            logger.info(
                f"[EARLY-STOP] No improvement for {self.patience} epochs. "
                f"Best AUC: {self.best_auc:.4f}")
            return True
        return False


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, frozen_bn_modules, dataloader, optimizer,
                    loc_w, cls_w, grad_clip):
    """
    Single-epoch training with gradient accumulation.

    BUG-B FIX: grad_clip is now 5.0 instead of 1.0, allowing natural
    gradient flow while preventing true gradient spikes.

    NOTE: gnorm logged to CSV is the PRE-CLIP total norm returned by
    clip_grad_norm_(). The actual applied gradient is clamped at grad_clip.
    This is the standard PyTorch convention and is informative for monitoring.
    """
    model.train()
    re_freeze_bn(frozen_bn_modules)   # defensive: ensure BN is frozen at epoch start

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
            # clip_grad_norm_ returns the pre-clip total norm
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip))
            step_gnorms.append(gnorm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

    if pending > 0:
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip))
        step_gnorms.append(gnorm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss  = float(np.mean(step_losses)) if step_losses else float('inf')
    avg_cls   = float(np.mean(step_cls))    if step_cls    else 0.0
    avg_loc   = float(np.mean(step_loc))    if step_loc    else 0.0
    avg_gnorm = float(np.mean(step_gnorms)) if step_gnorms else 0.0
    return avg_loss, avg_cls, avg_loc, avg_gnorm


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='HiFT fine-tuning — corrected v2.1')
    p.add_argument('--cfg',          default=DEFAULT_CFG)
    p.add_argument('--resume',       default=None)
    p.add_argument('--no-resume',    action='store_true')
    p.add_argument('--val-manifest', type=str, default=None)
    p.add_argument('--val-seqs',     type=int, default=20)
    p.add_argument('--max-val-frames', type=int, default=MAX_VAL_FRAMES,
                   help='Hard frame budget per validation run (default: 2000)')
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
    logger.warning("[VAL-MANIFEST] Not found — loss-only early stopping.")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

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
    model = ModelBuilder().cuda()

    total_epochs     = cfg.TRAIN.EPOCH
    base_lr          = cfg.TRAIN.BASE_LR
    warmup_ep        = cfg.TRAIN.LR_WARMUP.EPOCH
    trainable_layers = list(cfg.BACKBONE.TRAIN_LAYERS)

    set_backbone_trainable(model, trainable_layers)
    frozen_bn_modules = freeze_bn(model)

    optimizer = build_optimizer(model, base_lr)

    lr_cfg = cfg.TRAIN.LR
    if lr_cfg.TYPE == 'cosine':
        end_lr_frac = lr_cfg.KWARGS.end_lr / lr_cfg.KWARGS.start_lr
        scheduler   = build_cosine_scheduler(
            optimizer, total_epochs,
            warmup_epochs=warmup_ep,
            min_lr_frac=end_lr_frac)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    start_epoch = load_weights(model, optimizer, scheduler, args)

    model.train()
    re_freeze_bn(frozen_bn_modules)

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

    trainer = SafeTrainer(model, optimizer, scheduler,
                          _abs(cfg.TRAIN.SNAPSHOT_DIR))
    stopper  = EarlyStopper(patience=6, delta=1e-4, warmup=LOC_WEIGHT_FREEZE_EP)

    best_auc  = 0.0
    best_loss = float('inf')
    overall_t = time.time()

    log_csv = _abs(cfg.TRAIN.LOG_DIR) / 'training_log.csv'
    if not log_csv.exists():
        with open(log_csv, 'w') as f:
            f.write('epoch,train_loss,cls_loss,loc_loss,lr_head,'
                    'val_auc,val_precision,val_iou,'
                    'gnorm_preclip,loc_weight,val_frames\n')

    loc_shield_end = LOC_WEIGHT_FREEZE_EP + BACKBONE_LOC_SHIELD_EPOCHS

    logger.info(
        f"\n{'='*70}\n"
        f"HiFT FINE-TUNING v2.1 — LR Scheduler Fixed\n"
        f"  Epochs:          {start_epoch+1} → {total_epochs}\n"
        f"  LOC_WEIGHT:      0.0 (freeze {LOC_WEIGHT_FREEZE_EP} ep) → "
        f"{cfg.TRAIN.LOC_WEIGHT:.3f} (ramp {int(LOC_WEIGHT_RAMP_FRAC*100)}%)\n"
        f"  Loc freeze ep:   0-{LOC_WEIGHT_FREEZE_EP-1} "
        f"(LR peaks at ep {warmup_ep-1}, cools before loc starts)\n"
        f"  Backbone shield: ep {LOC_WEIGHT_FREEZE_EP}-{loc_shield_end-1} "
        f"(backbone LR × {BACKBONE_SHIELD_LR_MULT} during loc ramp-up)\n"
        f"  Base LR:         {base_lr:.2e}\n"
        f"  LR warmup:       {warmup_ep} epochs\n"
        f"  Grad clip:       {GRAD_CLIP_GLOBAL} (was 1.0 → raised to prevent over-clipping)\n"
        f"  Backbone:        {trainable_layers} (static freeze)\n"
        f"  BatchNorm:       FROZEN (re-frozen after every validation)\n"
        f"  VIDEOS/epoch:    {cfg.DATASET.VIDEOS_PER_EPOCH}\n"
        f"  Val frame cap:   {args.max_val_frames}\n"
        f"{'='*70}"
    )

    for epoch in range(start_epoch, total_epochs):
        if trainer.interrupted:
            break

        loc_w  = get_loc_weight(epoch, total_epochs, cfg.TRAIN.LOC_WEIGHT)
        cls_w  = cfg.TRAIN.CLS_WEIGHT

        dataset.pick = dataset.shuffle()

        epoch_t = time.time()
        
        # ── Training ────────────────────────────────────────────────────────
        avg_loss, avg_cls, avg_loc, avg_gnorm = train_one_epoch(
            model, frozen_bn_modules, dataloader, optimizer,
            loc_w, cls_w, GRAD_CLIP_GLOBAL)

        # ── Scheduler Step (ONCE, AFTER training) ─────────────────────────
        scheduler.step()
        cur_lrs = scheduler.get_last_lr()
        
        for idx, pg in enumerate(optimizer.param_groups):
            base_mult = pg.get('base_lr_mult', 1.0)
            is_backbone = pg.get('is_backbone', base_mult < 1.0)  # ← ADD THIS LINE
            shielded_mult = get_backbone_lr_mult(epoch, base_mult, is_backbone)  # ← PASS is_backbone
            pg['lr'] = cur_lrs[idx] * shielded_mult

        # Log HEAD LR (index -1 = last group = head)
        cur_lr = optimizer.param_groups[-1]['lr']

        logger.info(
            f"\n{'='*65}\n"
            f"Epoch [{epoch+1}/{total_epochs}] | "
            f"LOC_W: {loc_w:.4f} | "
            f"LR(head): {cur_lr:.2e} | "
            f"Backbone: {trainable_layers}"
            + (f" [SHIELDED x{BACKBONE_SHIELD_LR_MULT}]"
               if LOC_WEIGHT_FREEZE_EP <= epoch < loc_shield_end else "")
        )

        epoch_time = time.time() - epoch_t
        logger.info(
            f"Epoch {epoch+1} | {epoch_time:.0f}s | "
            f"loss={avg_loss:.4f} (cls={avg_cls:.4f} loc={avg_loc:.4f}) | "
            f"gnorm={avg_gnorm:.2f} | lr={cur_lr:.2e}"
        )
        if avg_gnorm > GRAD_CLIP_GLOBAL:
            logger.info(f"  [CLIP] Gradient was clipped "
                        f"({avg_gnorm:.2f} → {GRAD_CLIP_GLOBAL:.1f})")

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = None
        if args.val_seqs > 0 and val_manifest_path:
            val_metrics = run_validation(
                model, frozen_bn_modules,
                val_manifest_path, contest_dir,
                max_seqs=args.val_seqs,
                max_frames=args.max_val_frames)
            if val_metrics:
                logger.info(
                    f"[VAL] AUC={val_metrics['auc']:.4f} | "
                    f"Prec={val_metrics['precision']:.4f} | "
                    f"mIoU={val_metrics['mean_iou']:.4f} | "
                    f"frames={val_metrics['n_frames']}")
        else:
            re_freeze_bn(frozen_bn_modules)

        # ── Checkpoints ───────────────────────────────────────────────────────
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
            trainer.save(epoch, 'best.pth', extra)
            logger.info(f"  ★ New best val AUC: {best_auc:.4f}")

        # ── CSV log ───────────────────────────────────────────────────────────
        with open(log_csv, 'a') as f:
            f.write(
                f"{epoch+1},{avg_loss:.6f},{avg_cls:.6f},{avg_loc:.6f},"
                f"{cur_lr:.8f},"
                f"{val_metrics['auc']        if val_metrics else ''},"
                f"{val_metrics['precision']  if val_metrics else ''},"
                f"{val_metrics['mean_iou']   if val_metrics else ''},"
                f"{avg_gnorm:.4f},{loc_w:.4f},"
                f"{val_metrics['n_frames']   if val_metrics else ''}\n"
            )

        torch.cuda.empty_cache()

        val_auc_for_stop = val_metrics['auc'] if val_metrics else None
        if stopper.step(epoch, val_auc=val_auc_for_stop):
            logger.info(
                f"[EARLY-STOP] Triggered at epoch {epoch+1}. "
                f"Best val AUC: {best_auc:.4f}")
            break

        if trainer.interrupted:
            trainer.save(epoch, 'interrupted.pth', extra)
            break

    total_t = time.time() - overall_t
    logger.info(
        f"\n{'='*65}\n"
        f"Training complete!\n"
        f"  Duration:        {str(datetime.timedelta(seconds=int(total_t)))}\n"
        f"  Best val AUC:    {best_auc:.4f}\n"
        f"  Best train loss: {best_loss:.4f}\n"
        f"  Checkpoints:     {_abs(cfg.TRAIN.SNAPSHOT_DIR)}\n"
        f"{'='*65}"
    )


if __name__ == '__main__':
    main()