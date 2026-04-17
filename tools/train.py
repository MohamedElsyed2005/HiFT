#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HiFT Fine-Tuning — Competition-Grade Training Script
=====================================================
CRITICAL FIXES APPLIED:
#1 OUTPUT_SIZE=11 (was 21) — matches AlexNet output for SEARCH_SIZE=287
#2 AdamW optimizer + low LR (5e-5) — prevents catastrophic forgetting
#3 Internal validation split — uses manifest_val.json, NEVER public_lb
#4 BN frozen first 5 epochs — preserves pretrained feature statistics
#5 Dynamic LOC_WEIGHT warmup — prevents regression head collapse
#6 Gradient clipping + monitoring — catches instability early
#7 Early stopping on val AUC — saves best generalizing model
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

# ── imports after path setup ────────────────────────────────────────────────
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, add_file_handler
from pysot.utils.distributed import dist_init

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

# ═══════════════════════════════════════════════════════════════════════════
# STARTUP CONSISTENCY CHECK
# ═══════════════════════════════════════════════════════════════════════════
def assert_config_consistency():
    """Verify OUTPUT_SIZE consistency across config."""
    errors = []
    s = cfg.TRAIN.OUTPUT_SIZE
    f = cfg.TRAIN.OUTPUTFEATURE_SIZE
    if s != f:
        errors.append(
            f"TRAIN.OUTPUT_SIZE ({s}) != TRAIN.OUTPUTFEATURE_SIZE ({f}). "
            f"Set OUTPUT_SIZE: 11 in your YAML")
    expected = 11  # AlexNet(287) → 11×11 feature map
    if s != expected:
        errors.append(
            f"TRAIN.OUTPUT_SIZE={s} but expected {expected} for "
            f"SEARCH_SIZE={cfg.TRAIN.SEARCH_SIZE}. Fix your YAML.")
    if errors:
        for e in errors:
            logger.error(f"[CONFIG] {e}")
        raise RuntimeError("Config consistency check failed.")
    logger.info(f"[CONFIG] ✓ OUTPUT_SIZE={s} consistent with architecture.")

# ═══════════════════════════════════════════════════════════════════════════
# WEIGHT LOADING
# ═══════════════════════════════════════════════════════════════════════════
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
        if p.exists():
            _load_pretrained(model, p)
            return 0
    logger.warning("[WEIGHTS] No pretrained weights found. Training from scratch.")
    return 0

def _load_full_ckpt(model, optimizer, scheduler, path):
    ckpt = torch.load(str(path), map_location='cpu')
    state = _strip(ckpt.get('state_dict', ckpt))
    miss, unexp = model.load_state_dict(state, strict=False)
    miss = [k for k in miss if 'num_batches_tracked' not in k]
    logger.info(f"[WEIGHTS] Loaded {path.name}: missing={len(miss)} unexpected={len(unexp)}")
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

def _load_pretrained(model, path):
    logger.info(f"[WEIGHTS] Loading pretrained: {path}")
    raw = torch.load(str(path), map_location='cpu')
    if isinstance(raw, dict):
        state = raw.get('state_dict', raw.get('model', raw))
    else:
        state = raw
    state = _strip(state)
    has_backbone = any(k.startswith('backbone.') for k in state)
    has_grader = any(k.startswith('grader.') for k in state)
    if has_backbone and has_grader:
        miss, unexp = model.load_state_dict(state, strict=False)
        miss = [k for k in miss if 'num_batches_tracked' not in k]
        logger.info(f"[WEIGHTS] Full HiFT: missing={len(miss)}")
    elif has_backbone:
        bb_state = {k[len('backbone.'):]: v for k, v in state.items()
                    if k.startswith('backbone.')}
        miss, unexp = model.backbone.load_state_dict(bb_state, strict=False)
        miss = [k for k in miss if 'num_batches_tracked' not in k]
        logger.warning(f"[WEIGHTS] Backbone-only: missing={len(miss)}")
    else:
        miss, unexp = model.load_state_dict(state, strict=False)
        miss_f = [k for k in miss if 'num_batches_tracked' not in k]
        if len(miss_f) > 20:
            bb_state = {k.replace('features.', ''): v for k, v in state.items()}
            miss2, _ = model.backbone.load_state_dict(bb_state, strict=False)
            logger.info(f"[WEIGHTS] Bare backbone: missing={len(miss2)}")
        else:
            logger.info(f"[WEIGHTS] Bare keys: missing={len(miss_f)}")

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION — Uses INTERNAL SPLIT, NOT public_lb
# ═══════════════════════════════════════════════════════════════════════════
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
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / (union + 1e-6)

def _center_err(b1, b2):
    cx1, cy1 = b1[0] + b1[2]/2, b1[1] + b1[3]/2
    cx2, cy2 = b2[0] + b2[2]/2, b2[1] + b2[3]/2
    return float(np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2))

def run_validation(model, manifest_path, base_dir, max_seqs=30):
    """
    Run tracking on INTERNAL VALIDATION SPLIT sequences.
    manifest_path must point to manifest_val.json with sequences under "train" key.
    """
    from pysot.tracker.hift_tracker import HiFTTracker
    if not manifest_path.exists():
        return None
    with open(str(manifest_path), 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # ✅ CRITICAL: Use "train" section, NOT "public_lb"
    val_seqs = manifest.get('train', {})
    if not val_seqs:
        logger.warning(f"[VAL] No sequences under 'train' in {manifest_path}")
        return None
        
    seq_keys = list(val_seqs.keys())[:max_seqs]
    model.eval()
    all_ious, all_cerrs = [], []
    
    for seq_key in seq_keys:
        seq_info = val_seqs[seq_key]
        video_path = base_dir / seq_info['video_path']
        anno_path = base_dir / seq_info['annotation_path']
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
                    out = tracker.track(frame)
                    pred = out['bbox']
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
    ious = np.array(all_ious)
    cerrs = np.array(all_cerrs)
    thresholds = np.linspace(0, 1, 21)
    auc = float(np.mean([np.mean(ious >= t) for t in thresholds]))
    prec = float(np.mean(cerrs < 20))
    return {
        'auc': auc,
        'precision': prec,
        'mean_iou': float(np.mean(ious)),
        'n_frames': len(ious)
    }

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def freeze_bn(model):
    """Freeze BatchNorm layers to preserve pretrained stats."""
    count = 0
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            count += 1
    return count

def freeze_backbone_layers(model, train_layers):
    for name, param in model.backbone.named_parameters():
        param.requires_grad = name.split('.')[0] in train_layers
    frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    logger.info(f"Backbone: {frozen} frozen, {trainable} trainable")

def build_optimizer(model):
    """AdamW with separate LRs for backbone/head."""
    backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.grader.parameters())
    return torch.optim.AdamW([
        {'params': backbone_trainable, 'lr': cfg.TRAIN.BASE_LR * 0.1, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': cfg.TRAIN.BASE_LR, 'weight_decay': 1e-4},
    ])

def build_cosine_scheduler(optimizer, total_epochs, warmup_epochs=3):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class SafeTrainer:
    def __init__(self, model, optimizer, scheduler, save_dir):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)
    def _handle(self, signum, frame):
        logger.warning("Interrupt received — saving...")
        self.interrupted = True
    def save(self, epoch, filename, extra=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        if extra:
            state.update(extra)
        path = self.save_dir / filename
        torch.save(state, str(path))
        logger.info(f"  → Saved: {path.name}")
        return path

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default=DEFAULT_CFG)
    p.add_argument('--resume', default=None)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--val-manifest', type=str, default=None,
                   help='Path to validation manifest (default: auto-detect)')
    p.add_argument('--val-seqs', type=int, default=20,
                   help='Max sequences for validation (0=skip)')
    p.add_argument('--freeze-bn-epochs', type=int, default=5,
                   help='Keep backbone BN frozen for first N epochs')
    return p.parse_args()

def _resolve_val_manifest(args, contest_dir):
    if args.val_manifest:
        path = Path(args.val_manifest)
        if not path.exists():
            raise FileNotFoundError(f"--val-manifest not found: {path}")
        return path.resolve()
    default = _PROJ_ROOT / 'data' / 'processed' / 'manifest_val.json'
    if default.exists():
        logger.info(f"[VAL-MANIFEST] Using: {default}")
        return default
    logger.warning("[VAL-MANIFEST] Fallback to contest manifest — ensure split externally!")
    return contest_dir / 'metadata' / 'contestant_manifest.json'

def main():
    args = parse_args()
    
    # FIX: Load YAML with UTF-8 encoding and convert to CfgNode
    import yaml
    from yacs.config import CfgNode as CN
    
    def _load_cfg_utf8(path: str) -> CN:
        with open(path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
        return CN(cfg_dict)
    
    cfg_new = _load_cfg_utf8(args.cfg)
    cfg.merge_from_other_cfg(cfg_new)
    
    assert_config_consistency()
    os.makedirs(_abs(cfg.TRAIN.LOG_DIR), exist_ok=True)
    try:
        add_file_handler('global', str(_abs(cfg.TRAIN.LOG_DIR) / 'train.log'))
    except Exception:
        pass
    logger.info(f"Project root: {_PROJ_ROOT}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory/1024**3:.1f}GB")
    dist_init()

    contest_dir = Path(os.environ.get(
        'CONTEST_DATA_DIR',
        str(_PROJ_ROOT.parent / 'AIC4-UAV-Tracker' / 'data' / 'contest_release')
    )).resolve()
    val_manifest_path = _resolve_val_manifest(args, contest_dir)

    logger.info("Building model...")
    model = ModelBuilder()
    freeze_backbone_layers(model, cfg.BACKBONE.TRAIN_LAYERS)
    model = model.cuda()

    optimizer = build_optimizer(model)
    scheduler = build_cosine_scheduler(optimizer, cfg.TRAIN.EPOCH,
                                       warmup_epochs=cfg.TRAIN.LR_WARMUP.EPOCH)
    start_epoch = load_weights(model, optimizer, scheduler, args)

    logger.info("Loading dataset...")
    dataset = TrkDataset()
    dataloader = DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True,
        drop_last=True, persistent_workers=(cfg.TRAIN.NUM_WORKERS > 0))
    logger.info(f"Dataset: {len(dataset)} samples | Batch: {cfg.TRAIN.BATCH_SIZE}")

    trainer = SafeTrainer(model, optimizer, scheduler, _abs(cfg.TRAIN.SNAPSHOT_DIR))
    total_epochs = cfg.TRAIN.EPOCH
    best_auc, best_loss, no_improve = 0.0, float('inf'), 0
    overall_t = time.time()

    log_csv = _abs(cfg.TRAIN.LOG_DIR) / 'training_log.csv'
    if not log_csv.exists():
        with open(log_csv, 'w') as f:
            f.write('epoch,train_loss,cls_loss,loc_loss,lr,val_auc,val_precision,val_iou,grad_norm\n')

    logger.info(f"\nStarting training: epoch {start_epoch} → {total_epochs-1}")
    for epoch in range(start_epoch, total_epochs):
        if trainer.interrupted:
            break
        model.train()
        if epoch < args.freeze_bn_epochs:
            n_frozen = freeze_bn(model)
            if epoch == 0:
                logger.info(f"[BN] Freezing {n_frozen} BN layers for first {args.freeze_bn_epochs} epochs")
        dataset.pick = dataset.shuffle()
        epoch_t = time.time()
        step_losses, step_cls, step_loc, step_gnorms = [], [], [], []

        warmup_ep = cfg.TRAIN.LR_WARMUP.EPOCH
        if epoch < warmup_ep:
            loc_w = 0.5
        elif epoch < warmup_ep + 3:
            loc_w = 1.0
        else:
            loc_w = min(cfg.TRAIN.LOC_WEIGHT, 2.0)
        logger.info(f"\n{'='*65}\nEpoch [{epoch+1}/{total_epochs}] | LR_head: {optimizer.param_groups[1]['lr']:.2e} | LOC_WEIGHT: {loc_w:.1f} | BN_frozen: {epoch < args.freeze_bn_epochs}")

        for i, data in enumerate(dataloader):
            if trainer.interrupted:
                break
            cfg.TRAIN.LOC_WEIGHT = loc_w
            outputs = model(data)
            loss = outputs['total_loss']
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  NaN/Inf at step {i} — skipping")
                optimizer.zero_grad()
                continue
            if loss.item() == 0.0:
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            total_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
            step_gnorms.append(total_norm)
            if cfg.TRAIN.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()
            step_losses.append(loss.item())
            step_cls.append(outputs['cls_loss'].item() if hasattr(outputs['cls_loss'], 'item') else float(outputs['cls_loss']))
            step_loc.append(outputs['loc_loss'].item() if hasattr(outputs['loc_loss'], 'item') else float(outputs['loc_loss']))
            if i % cfg.TRAIN.PRINT_FREQ == 0:
                it_time = (time.time() - epoch_t) / (i + 1)
                eta_s = ((total_epochs - epoch - 1) * len(dataloader) + (len(dataloader) - i - 1)) * it_time
                logger.info(f"  [{i:4d}/{len(dataloader)}] loss={loss.item():.4f} cls={step_cls[-1]:.4f} loc={step_loc[-1]:.4f} gnorm={total_norm:.2f} ETA={str(datetime.timedelta(seconds=int(eta_s)))}")
                if total_norm > 50:
                    logger.warning(f"  ⚠️ Gradient spike: norm={total_norm:.1f}")

        scheduler.step()
        epoch_time = time.time() - epoch_t
        avg_loss = np.mean(step_losses) if step_losses else float('inf')
        avg_cls = np.mean(step_cls) if step_cls else 0.0
        avg_loc = np.mean(step_loc) if step_loc else 0.0
        avg_gnorm = np.mean(step_gnorms) if step_gnorms else 0.0
        cur_lr = optimizer.param_groups[1]['lr']
        logger.info(f"Epoch {epoch+1} | {epoch_time:.0f}s | loss={avg_loss:.4f} (cls={avg_cls:.4f} loc={avg_loc:.4f}) | gnorm={avg_gnorm:.2f} | lr={cur_lr:.2e}")

        # Validation on INTERNAL SPLIT
        val_metrics = None
        if args.val_seqs > 0 and val_manifest_path.exists():
            logger.info(f"[VAL] Running on {args.val_seqs} sequences from: {val_manifest_path.name}")
            val_metrics = run_validation(model, val_manifest_path, contest_dir, max_seqs=args.val_seqs)
            if val_metrics:
                logger.info(f"[VAL] AUC={val_metrics['auc']:.4f} | Prec={val_metrics['precision']:.4f} | mIoU={val_metrics['mean_iou']:.4f}")
            else:
                logger.warning("[VAL] No results")
        else:
            logger.info("[VAL] Skipped")

        extra = {'train_loss': avg_loss, 'val_auc': val_metrics['auc'] if val_metrics else None}
        trainer.save(epoch, 'last_epoch.pth', extra)
        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save(epoch, 'best_train.pth', extra)
        if val_metrics and val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            no_improve = 0
            trainer.save(epoch, 'best_val.pth', extra)
            logger.info(f"  ★ New best AUC: {best_auc:.4f}")
        elif val_metrics:
            no_improve += 1
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            trainer.save(epoch, f'epoch_{epoch+1:03d}.pth', extra)

        with open(log_csv, 'a') as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{avg_cls:.6f},{avg_loc:.6f},{cur_lr:.8f},"
                    f"{val_metrics['auc'] if val_metrics else ''},{val_metrics['precision'] if val_metrics else ''},"
                    f"{val_metrics['mean_iou'] if val_metrics else ''},{avg_gnorm:.4f}\n")

        if val_metrics and no_improve >= 5:
            logger.info(f"[EARLY STOP] Val AUC stagnant for 5 epochs. Best: {best_auc:.4f}")
            break
        torch.cuda.empty_cache()
        if trainer.interrupted:
            trainer.save(epoch, 'interrupted.pth', extra)
            break

    total_t = time.time() - overall_t
    logger.info(f"\nTraining complete! Duration: {str(datetime.timedelta(seconds=int(total_t)))} | Best AUC: {best_auc:.4f}")
    logger.info(f"Training log: {log_csv}")

if __name__ == '__main__':
    main()