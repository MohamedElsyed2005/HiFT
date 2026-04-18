#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/train.py  (FIXED)
=======================
Changes from previous version:

FIX-LOCHWEIGHT: get_loc_weight start changed from 0.3 → 0.1
  With only 202 training sequences, starting the regression head weight at 0.3
  is already too high — the classification head hasn't found the target region
  yet, so the regression loss is random, producing noisy gradients that spike
  total loss at epoch 6.  Starting at 0.1 gives the classification head 3-5
  epochs to stabilise before regression takes weight.

FIX-VAL-MANIFEST: _resolve_val_manifest now also checks for 'val.json'
  preprocess_data.py (fixed) writes val.json alongside train.json.
  The previous code only looked for manifest_val.json (wrong format).

All other fixes (A–H) from the previous version are retained.
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

PATIENCE_VAL      = 8
PATIENCE_LOSS     = 5
LOSS_DELTA_THRESH = 5e-4
ACCUM_STEPS       = 4


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


def assert_config_consistency():
    errors = []
    s = cfg.TRAIN.OUTPUT_SIZE
    f = cfg.TRAIN.OUTPUTFEATURE_SIZE
    if s != f:
        errors.append(
            f"TRAIN.OUTPUT_SIZE ({s}) != TRAIN.OUTPUTFEATURE_SIZE ({f}).")
    if s != 11:
        errors.append(
            f"TRAIN.OUTPUT_SIZE={s} but expected 11 for SEARCH_SIZE={cfg.TRAIN.SEARCH_SIZE}.")
    if errors:
        for e in errors:
            logger.error(f"[CONFIG] {e}")
        raise RuntimeError("Config consistency check failed.")
    logger.info(f"[CONFIG] OUTPUT_SIZE={s} consistent ✓")


# ── LOC_WEIGHT SCHEDULE (FIXED: start=0.1 not 0.3) ──────────────────────────
def get_loc_weight(epoch: int, total_epochs: int, target: float,
                   start: float = 0.1) -> float:
    """
    Linear ramp from `start` → `target` over the FULL training duration.

    FIXED: start changed from 0.3 → 0.1
    With only ~200 sequences, starting at 0.3 caused the regression head to
    chase random targets before the classification head had stabilised (visible
    as the +17% total loss spike at epoch 6 in training logs).
    Starting at 0.1 gives the classification head ~3 epochs to find targets.
    """
    progress = min(epoch / max(total_epochs - 1, 1), 1.0)
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
    Evaluate on the internal validation split (val.json).
    val.json is the crop-annotation JSON — we look up the original video
    paths from the contestant_manifest for cap.read().
    """
    from pysot.tracker.hift_tracker import HiFTTracker

    if not manifest_path or not Path(manifest_path).exists():
        return None

    # val.json is a crop-annotation file: {seq_key: {"0": {frame: bbox}}}
    # We need to find the original videos; look in the contestant manifest.
    contest_manifest_path = base_dir / 'metadata' / 'contestant_manifest.json'
    if not contest_manifest_path.exists():
        logger.warning(f"[VAL] Contest manifest not found: {contest_manifest_path}")
        return None

    with open(str(contest_manifest_path), 'r', encoding='utf-8') as f:
        contest_manifest = json.load(f)
    all_seqs = {**contest_manifest.get('train', {}),
                **contest_manifest.get('public_lb', {})}

    with open(str(manifest_path), 'r', encoding='utf-8') as f:
        val_annos = json.load(f)

    seq_keys = list(val_annos.keys())
    if max_seqs > 0:
        rng = np.random.default_rng(seed=42)
        rng.shuffle(seq_keys)
        seq_keys = seq_keys[:max_seqs]

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
        gt        = _load_gt(anno_path)
        if len(gt) < 2:
            continue
        init_bbox = _parse_init_bbox(anno_path)
        if init_bbox is None:
            continue
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
            all_ious.append(1.0)
            all_cerrs.append(0.0)
            last_pred = init_bbox
            for frame_idx in range(1, len(gt)):
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    out       = tracker.track(frame)
                    pred      = out['bbox']
                    last_pred = pred
                except Exception:
                    pred = last_pred
                all_ious.append(_iou_xywh(pred, gt[frame_idx]))
                all_cerrs.append(_center_err(pred, gt[frame_idx]))
            cap.release()
        except Exception as e:
            logger.warning(f"[VAL] Error on {seq_key}: {e}")
            continue

    if not all_ious:
        return None

    ious  = np.array(all_ious)
    cerrs = np.array(all_cerrs)
    thresholds = np.linspace(0, 1, 21)
    auc  = float(np.mean([np.mean(ious >= t) for t in thresholds]))
    prec = float(np.mean(cerrs < 20))
    return {'auc': auc, 'precision': prec,
            'mean_iou': float(np.mean(ious)), 'n_frames': len(ious)}


# ── Model / Optimizer / Scheduler ─────────────────────────────────────────────
def freeze_bn(model):
    count = 0
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            count += 1
    return count


def freeze_backbone_layers(model, train_layers):
    for name, param in model.backbone.named_parameters():
        param.requires_grad = name.split('.')[0] in train_layers
    frozen    = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    logger.info(f"Backbone: {frozen} frozen, {trainable} trainable")


def build_optimizer(model):
    backbone_trainable = [p for p in model.backbone.parameters()
                          if p.requires_grad]
    head_params = list(model.grader.parameters())
    return torch.optim.AdamW([
        {'params': backbone_trainable, 'lr': cfg.TRAIN.BASE_LR * 0.1,
         'weight_decay': 1e-4},
        {'params': head_params,        'lr': cfg.TRAIN.BASE_LR,
         'weight_decay': 5e-4},
    ])


def build_cosine_scheduler(optimizer, total_epochs,
                            warmup_epochs: int = 3,
                            hold_epochs: int = 3):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        if epoch < warmup_epochs + hold_epochs:
            return 1.0
        decay_start = warmup_epochs + hold_epochs
        decay_span  = max(total_epochs - decay_start, 1)
        progress    = (epoch - decay_start) / decay_span
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── SafeTrainer ────────────────────────────────────────────────────────────────
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
        logger.warning("Interrupt — saving checkpoint.")
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
    def __init__(self, patience_val=PATIENCE_VAL, patience_loss=PATIENCE_LOSS,
                 delta=LOSS_DELTA_THRESH, warmup=5):
        self.patience_val  = patience_val
        self.patience_loss = patience_loss
        self.delta         = delta
        self.warmup        = warmup
        self.best_auc      = -1.0
        self.best_loss     = float('inf')
        self.no_improve    = 0
        self._loss_window  = deque(maxlen=3)

    def step(self, epoch, val_auc=None, train_loss=None):
        if epoch < self.warmup:
            return False
        if val_auc is not None:
            if val_auc > self.best_auc + self.delta:
                self.best_auc   = val_auc
                self.no_improve = 0
                return False
            self.no_improve += 1
            logger.info(f"[EARLY-STOP] No val improvement "
                        f"({self.no_improve}/{self.patience_val})")
            return self.no_improve >= self.patience_val
        if train_loss is not None:
            self._loss_window.append(train_loss)
            if len(self._loss_window) < self._loss_window.maxlen:
                return False
            rolling = float(np.mean(self._loss_window))
            if self.best_loss - rolling > self.delta:
                self.best_loss  = rolling
                self.no_improve = 0
                return False
            self.no_improve += 1
            logger.info(f"[EARLY-STOP] No loss improvement "
                        f"({self.no_improve}/{self.patience_loss})")
            return self.no_improve >= self.patience_loss
        return False


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, loc_w, cls_w, grad_clip):
    model.train()
    step_losses, step_cls, step_loc, step_gnorms = [], [], [], []
    optimizer.zero_grad(set_to_none=True)
    pending = 0

    for i, data in enumerate(dataloader):
        outputs = model(data, loc_weight=loc_w, cls_weight=cls_w)
        loss    = outputs['total_loss'] / ACCUM_STEPS
        loss.backward()
        pending += 1

        step_losses.append(outputs['total_loss'].item())
        step_cls.append(float(outputs['cls_loss']))
        step_loc.append(float(outputs['loc_loss']))

        if pending == ACCUM_STEPS:
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip)
            step_gnorms.append(float(gnorm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

    if pending > 0:
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        step_gnorms.append(float(gnorm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss  = float(np.mean(step_losses)) if step_losses else float('inf')
    avg_cls   = float(np.mean(step_cls))   if step_cls   else 0.0
    avg_loc   = float(np.mean(step_loc))   if step_loc   else 0.0
    avg_gnorm = float(np.mean(step_gnorms)) if step_gnorms else 0.0
    return avg_loss, avg_cls, avg_loc, avg_gnorm


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg',          default=DEFAULT_CFG)
    p.add_argument('--resume',       default=None)
    p.add_argument('--no-resume',    action='store_true')
    p.add_argument('--val-manifest', type=str, default=None)
    p.add_argument('--val-seqs',     type=int, default=15)
    p.add_argument('--freeze-bn-epochs', type=int, default=8)
    return p.parse_args()


def _resolve_val_manifest(args):
    """
    FIXED: now checks val.json (written by fixed preprocess_data.py)
    in addition to manifest_val.json (legacy name).
    """
    if args.val_manifest:
        path = Path(args.val_manifest)
        if not path.exists():
            raise FileNotFoundError(f"--val-manifest not found: {path}")
        return path.resolve()
    # Try val.json first (fixed preprocess writes this)
    for candidate_name in ['val.json', 'manifest_val.json']:
        default = _PROJ_ROOT / 'data' / 'processed' / candidate_name
        if default.exists():
            logger.info(f"[VAL-MANIFEST] Using: {default}")
            return default
    logger.warning("[VAL-MANIFEST] Not found — using train-loss early stopping.")
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

    logger.info(f"Project root: {_PROJ_ROOT}")
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

    logger.info("Building model...")
    model = ModelBuilder()
    freeze_backbone_layers(model, cfg.BACKBONE.TRAIN_LAYERS)
    model = model.cuda()

    optimizer = build_optimizer(model)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_epochs=cfg.TRAIN.EPOCH,
        warmup_epochs=cfg.TRAIN.LR_WARMUP.EPOCH,
        hold_epochs=3,
    )

    start_epoch = load_weights(model, optimizer, scheduler, args)

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

    trainer      = SafeTrainer(model, optimizer, scheduler,
                               _abs(cfg.TRAIN.SNAPSHOT_DIR))
    stopper      = EarlyStopper(warmup=max(5, start_epoch + 3))
    total_epochs = cfg.TRAIN.EPOCH
    best_auc     = 0.0
    best_loss    = float('inf')
    overall_t    = time.time()

    log_csv = _abs(cfg.TRAIN.LOG_DIR) / 'training_log.csv'
    if not log_csv.exists():
        with open(log_csv, 'w') as f:
            f.write('epoch,train_loss,cls_loss,loc_loss,lr,'
                    'val_auc,val_precision,val_iou,grad_norm,loc_weight\n')

    logger.info(f"Training: epoch {start_epoch+1} → {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        if trainer.interrupted:
            break

        if epoch < args.freeze_bn_epochs:
            n_frozen = freeze_bn(model)
            if epoch == 0:
                logger.info(
                    f"[BN] Freezing {n_frozen} BN layers "
                    f"(unfreeze at epoch {args.freeze_bn_epochs})")
        elif epoch == args.freeze_bn_epochs:
            logger.info("[BN] Unfreezing backbone BN layers.")

        dataset.pick = dataset.shuffle()

        # FIXED: start=0.1 (was 0.3)
        loc_w  = get_loc_weight(epoch, total_epochs, cfg.TRAIN.LOC_WEIGHT,
                                start=0.1)
        cls_w  = cfg.TRAIN.CLS_WEIGHT
        cur_lr = optimizer.param_groups[1]['lr']

        logger.info(
            f"\n{'='*65}\n"
            f"Epoch [{epoch+1}/{total_epochs}] | "
            f"LOC_WEIGHT: {loc_w:.3f} | LR: {cur_lr:.2e}"
        )

        epoch_t = time.time()
        avg_loss, avg_cls, avg_loc, avg_gnorm = train_one_epoch(
            model, dataloader, optimizer, loc_w, cls_w,
            grad_clip=cfg.TRAIN.GRAD_CLIP)
        scheduler.step()

        epoch_time = time.time() - epoch_t
        logger.info(
            f"Epoch {epoch+1} | {epoch_time:.0f}s | "
            f"loss={avg_loss:.4f} (cls={avg_cls:.4f} loc={avg_loc:.4f}) | "
            f"gnorm={avg_gnorm:.2f} | lr={cur_lr:.2e}"
        )

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

        extra = {
            'train_loss': avg_loss,
            'val_auc':    val_metrics['auc'] if val_metrics else None,
            'loc_weight': loc_w,
        }
        trainer.save(epoch, 'latest.pth', extra)

        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save(epoch, 'best_train.pth', extra)

        if val_metrics and val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            trainer.save(epoch, 'best_val.pth', extra)

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

        val_auc_for_stop = val_metrics['auc'] if val_metrics else None
        if stopper.step(epoch, val_auc=val_auc_for_stop,
                        train_loss=avg_loss):
            logger.info(
                f"[EARLY-STOP] Epoch {epoch+1}. "
                f"Best val AUC: {best_auc:.4f}  "
                f"Best train loss: {best_loss:.4f}")
            break

        if trainer.interrupted:
            trainer.save(epoch, 'interrupted.pth', extra)
            break

    total_t = time.time() - overall_t
    logger.info(
        f"\nDone! Duration: "
        f"{str(datetime.timedelta(seconds=int(total_t)))} | "
        f"Best val AUC: {best_auc:.4f} | "
        f"Best train loss: {best_loss:.4f}"
    )


if __name__ == '__main__':
    main()