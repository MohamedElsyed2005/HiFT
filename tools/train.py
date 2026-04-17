#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HiFT Fine-Tuning Script — FIXED pretrained weight loading

=============================================================================
PRETRAINED WEIGHT LOADING BUGS FIXED IN THIS VERSION:

BUG 1 — Path resolution used cwd-relative os.path.exists().
  OLD: os.path.exists(cfg.BACKBONE.PRETRAINED)  ← fails unless cwd==proj root
  FIX: All paths resolved to absolute via _abs() anchored to this file's
       location, identical to the pattern used in dataset.py.

BUG 2 — first.pth (full HiFT checkpoint) was loaded into model.backbone only.
  OLD: model.backbone.load_state_dict(pretrained, strict=False)
       → HiFT head (model.grader) ALWAYS randomly initialized
  FIX: _load_hift_pretrained() loads into the full model. It detects whether
       the checkpoint has 'grader.*' keys and handles all three formats:
         A) full model keys  (backbone.* + grader.*)  → load into full model
         B) backbone keys only (backbone.*)           → load backbone portion
         C) bare AlexNet keys (layer1.*, layer2.*, …) → load into backbone

BUG 3 — alexnet-bn.pth was never referenced by any config key or code path.
  FIX: Added explicit fallback search for pretrained/alexnet-bn.pth after
       first.pth is not found. _load_backbone_only() strips common prefixes
       ('module.', 'features.') to handle all standard AlexNet checkpoint
       formats.

BUG 4 — snapshot/latest.pth was auto-detected even when the intent was to
  start a fresh fine-tuning run from first.pth (e.g. after a config change).
  FIX: Added --no-resume flag. Without it, auto-resume still works as before.
       With it, the code skips straight to pretrained weight loading.

BUG 5 — Warning message used a hardcoded wrong path string.
  OLD: logger.warning(f"No pretrained weights found at {cfg.BACKBONE.PRETRAINED}.")
       (resolves against cwd, gives a confusing path in the message)
  FIX: Warning shows ALL candidate paths that were tried (absolute), so it
       is immediately clear which directories to put the files in.
=============================================================================
"""

import os
import sys
import time
import datetime
import signal
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Project root: this file lives at <root>/tools/train.py ─────────────────
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

DEFAULT_CFG = str(_PROJ_ROOT / 'configs' / 'hiFT_finetune.yaml')


def _abs(path_str: str) -> Path:
    """
    Resolve a config path to an absolute Path anchored at the project root.
    This is the SAME pattern used in pysot/datasets/dataset.py (_abs_path).
    """
    p = Path(path_str)
    return p if p.is_absolute() else (_PROJ_ROOT / p).resolve()


from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, add_file_handler
from pysot.utils.distributed import dist_init


def get_logger(name='global', level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger('global', logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='HiFT Fine-tuning')
    parser.add_argument('--cfg', type=str, default=DEFAULT_CFG)
    parser.add_argument('--resume', type=str, default=None,
                        help='Explicit path to checkpoint to resume from')
    # FIX BUG 4: allow skipping auto-resume to force pretrained load
    parser.add_argument('--no-resume', action='store_true',
                        help='Skip auto-detection of snapshot/latest.pth '
                             'and load from pretrained weights instead. '
                             'Use this when starting a fresh fine-tuning run.')
    return parser.parse_args()


# =============================================================================
# WEIGHT LOADING — FIXED
# =============================================================================

def _log_key_stats(path: Path, missing: list, unexpected: list, scope: str = 'model'):
    """Log missing/unexpected key statistics with useful context."""
    # num_batches_tracked mismatches are always benign — filter them out
    missing = [k for k in missing if 'num_batches_tracked' not in k]
    if missing:
        logger.warning(
            f"[WEIGHTS] {scope} ← {path.name}: "
            f"{len(missing)} missing keys. First 5: {missing[:5]}"
        )
    if unexpected:
        logger.info(
            f"[WEIGHTS] {scope} ← {path.name}: "
            f"{len(unexpected)} unexpected/extra keys (ignored — normal for partial loads)"
        )
    if not missing and not unexpected:
        logger.info(f"[WEIGHTS] {scope} ← {path.name}: perfect key match ✓")


def _strip_prefixes(state_dict: dict) -> dict:
    """Remove 'module.' (DataParallel) prefix."""
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def _load_full_checkpoint(model, optimizer, path: Path) -> int:
    """
    Load a full training checkpoint (state_dict + optimizer + epoch).
    Returns start_epoch.
    """
    logger.info(f"[WEIGHTS] Loading full checkpoint: {path}")
    ckpt = torch.load(str(path), map_location='cpu')

    state = ckpt.get('state_dict', ckpt)
    state = _strip_prefixes(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    _log_key_stats(path, missing, unexpected, scope='full model')

    if 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            for st in optimizer.state.values():
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        st[k] = v.cuda()
            logger.info("[WEIGHTS] Optimizer state restored.")
        except Exception as e:
            logger.warning(f"[WEIGHTS] Could not restore optimizer state: {e}")

    start_epoch = ckpt.get('epoch', -1) + 1
    logger.info(f"[WEIGHTS] Resuming from epoch {start_epoch}.")
    return start_epoch


def _load_hift_pretrained(model, path: Path):
    """
    Load first.pth — a full HiFT pretrained checkpoint — into the model.

    Handles all observed checkpoint formats:
      Format A: keys prefixed with 'backbone.' and 'grader.'
                → full model load (backbone + HiFT head both initialized)
      Format B: keys prefixed with 'backbone.' only (no grader keys)
                → backbone-only load; head trains from scratch
      Format C: bare AlexNet keys (layer1.*, layer2.*, …)
                → treated as backbone-only weights
    """
    logger.info(f"[WEIGHTS] Loading HiFT pretrained checkpoint: {path}")
    raw = torch.load(str(path), map_location='cpu')

    # Unwrap common wrappers
    if isinstance(raw, dict):
        if 'state_dict' in raw:
            state = raw['state_dict']
        elif 'model' in raw:
            state = raw['model']
        else:
            state = raw
    else:
        raise ValueError(f"Unexpected checkpoint type {type(raw)} in {path}")

    state = _strip_prefixes(state)

    has_backbone_prefix = any(k.startswith('backbone.') for k in state)
    has_grader_prefix   = any(k.startswith('grader.')   for k in state)

    if has_backbone_prefix and has_grader_prefix:
        # ── Format A: full HiFT checkpoint ───────────────────────────────
        logger.info("[WEIGHTS] Format A detected: backbone + grader keys present.")
        logger.info("[WEIGHTS] Loading into full model (backbone + HiFT head).")
        missing, unexpected = model.load_state_dict(state, strict=False)
        _log_key_stats(path, missing, unexpected, scope='full model')

    elif has_backbone_prefix and not has_grader_prefix:
        # ── Format B: backbone keys only ─────────────────────────────────
        logger.warning(
            "[WEIGHTS] Format B detected: only backbone.* keys found in first.pth.\n"
            "         The HiFT head (grader) is NOT in this checkpoint and will\n"
            "         train from scratch. If this is unexpected, check first.pth."
        )
        backbone_state = {k[len('backbone.'):]: v
                          for k, v in state.items()
                          if k.startswith('backbone.')}
        missing, unexpected = model.backbone.load_state_dict(
            backbone_state, strict=False)
        _log_key_stats(path, missing, unexpected, scope='backbone')

    else:
        # ── Format C: bare weights (no backbone./grader. prefixes) ───────
        # Determine whether it looks like a full HiFT model or bare AlexNet
        has_conv1_grader = any('conv1' in k and 'layer' not in k for k in state)
        logger.info(
            "[WEIGHTS] Format C detected: no backbone./grader. prefixes.\n"
            "         Attempting full model load first, then backbone fallback."
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        # If huge numbers of keys are missing, this is likely a backbone-only file
        missing_filtered = [k for k in missing if 'num_batches_tracked' not in k]
        if len(missing_filtered) > 20:
            logger.warning(
                f"[WEIGHTS] Full model load left {len(missing_filtered)} missing keys.\n"
                "         Retrying as bare backbone load into model.backbone."
            )
            # Try stripping 'features.' prefix common in torchvision AlexNet checkpoints
            backbone_state = {
                k.replace('features.', ''): v for k, v in state.items()
            }
            missing2, unexpected2 = model.backbone.load_state_dict(
                backbone_state, strict=False)
            _log_key_stats(path, missing2, unexpected2, scope='backbone (bare)')
        else:
            _log_key_stats(path, missing, unexpected, scope='full model (bare)')


def _load_backbone_only(model, path: Path):
    """
    Load alexnet-bn.pth (bare AlexNet weights) into model.backbone.
    Handles torchvision-style 'features.*' prefix and bare layer keys.
    """
    logger.info(f"[WEIGHTS] Loading backbone-only weights: {path}")
    raw = torch.load(str(path), map_location='cpu')

    if isinstance(raw, dict) and 'state_dict' in raw:
        state = raw['state_dict']
    elif isinstance(raw, dict) and 'model' in raw:
        state = raw['model']
    else:
        state = raw

    state = _strip_prefixes(state)
    # Strip 'features.' prefix (torchvision AlexNet format)
    state = {k.replace('features.', ''): v for k, v in state.items()}

    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    _log_key_stats(path, missing, unexpected, scope='backbone')
    logger.info(
        "[WEIGHTS] ✓ Backbone loaded from alexnet-bn.pth.\n"
        "         The HiFT head (grader) will train from scratch."
    )


def load_weights(model, optimizer, args) -> int:
    """
    Master weight-loading function. Returns start_epoch (int).

    Priority:
      1. --resume <path>           (explicit, highest priority)
      2. snapshot/latest.pth       (auto-resume, skipped if --no-resume)
      3. pretrained/first.pth      (full HiFT pretrained)
      4. pretrained/alexnet-bn.pth (backbone-only fallback)
      5. Nothing → train from scratch (logged loudly)
    """

    # ── 1. Explicit --resume ─────────────────────────────────────────────────
    if args.resume:
        path = _abs(args.resume)
        if not path.exists():
            raise FileNotFoundError(
                f"[WEIGHTS] --resume path not found: {path}\n"
                f"         Check the path and try again.")
        return _load_full_checkpoint(model, optimizer, path)

    # ── 2. Auto-resume from snapshot/latest.pth ──────────────────────────────
    if not args.no_resume:
        auto_path = _abs(cfg.TRAIN.SNAPSHOT_DIR) / 'latest.pth'
        if auto_path.exists():
            logger.info(
                f"[WEIGHTS] Auto-resuming from: {auto_path}\n"
                "         (run with --no-resume to skip this and load pretrained instead)"
            )
            return _load_full_checkpoint(model, optimizer, auto_path)
    else:
        logger.info("[WEIGHTS] --no-resume set: skipping snapshot/latest.pth auto-detection.")

    # ── 3. Full HiFT pretrained checkpoint (first.pth) ───────────────────────
    hift_candidates = [
        _abs(cfg.BACKBONE.PRETRAINED),           # from config/YAML
        _PROJ_ROOT / 'pretrained' / 'first.pth',
        _PROJ_ROOT / 'pretrained_models' / 'first.pth',
        _PROJ_ROOT / 'snapshot' / 'first.pth',
    ]
    # Deduplicate while preserving order
    seen = set()
    hift_candidates = [
        p for p in hift_candidates
        if not (str(p) in seen or seen.add(str(p)))
    ]

    for path in hift_candidates:
        if path.exists():
            _load_hift_pretrained(model, path)
            logger.info(
                f"[WEIGHTS] ✓ HiFT pretrained loaded from: {path}\n"
                "         Fine-tuning mode active."
            )
            return 0  # epoch 0, not a resume

    logger.warning(
        "[WEIGHTS] Full HiFT checkpoint (first.pth) not found. Tried:\n"
        + "\n".join(f"         {p}" for p in hift_candidates)
    )

    # ── 4. Backbone-only fallback: alexnet-bn.pth ────────────────────────────
    backbone_candidates = [
        _PROJ_ROOT / 'pretrained' / 'alexnet-bn.pth',
        _PROJ_ROOT / 'pretrained_models' / 'alexnet-bn.pth',
    ]

    for path in backbone_candidates:
        if path.exists():
            _load_backbone_only(model, path)
            logger.info(
                f"[WEIGHTS] ✓ Backbone (AlexNet) loaded from: {path}\n"
                "         HiFT head will train from scratch."
            )
            return 0

    # ── 5. No weights found ──────────────────────────────────────────────────
    logger.warning(
        "\n"
        "  ╔══════════════════════════════════════════════════════════════╗\n"
        "  ║  WARNING: NO PRETRAINED WEIGHTS FOUND                       ║\n"
        "  ║  Model is initializing ENTIRELY FROM SCRATCH.               ║\n"
        "  ║                                                              ║\n"
        "  ║  To fine-tune properly, place your files here:              ║\n"
        f"  ║    {str(_PROJ_ROOT / 'pretrained' / 'first.pth'):<54}║\n"
        f"  ║    {str(_PROJ_ROOT / 'pretrained' / 'alexnet-bn.pth'):<54}║\n"
        "  ║                                                              ║\n"
        "  ║  Then re-run with: python tools/train.py --no-resume        ║\n"
        "  ╚══════════════════════════════════════════════════════════════╝\n"
    )
    return 0


# =============================================================================
# REST OF train.py (unchanged except for the weight loading call)
# =============================================================================

class SafeTrainer:
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.save_dir = _abs(cfg.TRAIN.SNAPSHOT_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interrupted = False
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.warning("Interrupt received! Saving before exit...")
        self.interrupted = True

    def save_checkpoint(self, epoch, filename, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        path = self.save_dir / filename
        torch.save(state, str(path))
        logger.info(f"Saved: {path}")
        if is_best:
            best_path = self.save_dir / 'best.pth'
            torch.save(state, str(best_path))
            logger.info(f"New BEST model: {best_path}")


def freeze_backbone_layers(model, train_layers):
    """Freeze all backbone layers EXCEPT those listed in train_layers."""
    for name, param in model.backbone.named_parameters():
        layer_name = name.split('.')[0]
        param.requires_grad = layer_name in train_layers

    frozen   = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    logger.info(f"Backbone frozen params: {frozen} | trainable: {trainable}")


def main():
    args = parse_args()
    cfg.merge_from_file(args.cfg)

    os.makedirs(_abs(cfg.TRAIN.LOG_DIR), exist_ok=True)
    try:
        add_file_handler('global', str(_abs(cfg.TRAIN.LOG_DIR) / 'train.log'))
    except Exception:
        pass

    logger.info(f"Config: {args.cfg}")
    logger.info(f"Project root: {_PROJ_ROOT}")
    if torch.cuda.is_available():
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    rank, world_size = dist_init()

    # ── Build model ─────────────────────────────────────────────────────────
    logger.info("Building model...")
    model = ModelBuilder()
    freeze_backbone_layers(model, cfg.BACKBONE.TRAIN_LAYERS)
    model = model.cuda()

    # ── Optimizer ────────────────────────────────────────────────────────────
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.grader.parameters())

    optimizer = torch.optim.SGD(
        [
            {'params': backbone_params,
             'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR},
            {'params': head_params,
             'lr': cfg.TRAIN.BASE_LR}
        ],
        momentum=cfg.TRAIN.MOMENTUM,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )

    # ── Load weights (FIXED) ─────────────────────────────────────────────────
    start_epoch = load_weights(model, optimizer, args)

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    lr_scheduler = build_lr_scheduler(
        optimizer,
        epochs=cfg.TRAIN.EPOCH,
        last_epoch=start_epoch - 1
    )

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    logger.info("Loading dataset...")
    dataset = TrkDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.TRAIN.NUM_WORKERS > 0),
    )
    logger.info(
        f"Dataset: {len(dataset)} samples | "
        f"Batch: {cfg.TRAIN.BATCH_SIZE} | "
        f"Steps/epoch: {len(dataloader)}"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    trainer      = SafeTrainer(model, optimizer, cfg)
    total_epochs = cfg.TRAIN.EPOCH
    best_loss    = float('inf')
    overall_start = time.time()

    logger.info(f"Starting training: epoch {start_epoch} → {total_epochs - 1}")

    for epoch in range(start_epoch, total_epochs):
        if trainer.interrupted:
            break

        dataset.pick = dataset.shuffle()
        model.train()
        epoch_start  = time.time()
        total_losses = []
        cls_losses   = []
        loc_losses   = []

        logger.info(
            f"\n{'='*60}\n"
            f"Epoch [{epoch+1}/{total_epochs}] | "
            f"LR: {lr_scheduler.get_cur_lr():.6f}"
        )

        for i, data in enumerate(dataloader):
            if trainer.interrupted:
                break

            outputs = model(data)
            loss = outputs['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  NaN/Inf loss at step {i}, skipping batch")
                optimizer.zero_grad()
                continue

            if loss.item() == 0.0:
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()

            if cfg.TRAIN.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.TRAIN.GRAD_CLIP)

            optimizer.step()

            total_losses.append(loss.item())
            cls_losses.append(
                outputs['cls_loss'].item()
                if not isinstance(outputs['cls_loss'], float)
                else outputs['cls_loss'])
            loc_losses.append(
                outputs['loc_loss'].item()
                if not isinstance(outputs['loc_loss'], float)
                else outputs['loc_loss'])

            if i % cfg.TRAIN.PRINT_FREQ == 0:
                lr        = lr_scheduler.get_cur_lr()
                iter_time = (time.time() - epoch_start) / (i + 1)
                remaining = (
                    (total_epochs - epoch - 1) * len(dataloader)
                    + (len(dataloader) - i - 1)
                ) * iter_time
                logger.info(
                    f"  [{i:4d}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(cls={cls_losses[-1]:.4f} loc={loc_losses[-1]:.4f}) "
                    f"LR: {lr:.6f} "
                    f"ETA: {str(datetime.timedelta(seconds=int(remaining)))}"
                )

        epoch_time = time.time() - epoch_start
        lr_scheduler.step()

        avg_loss = sum(total_losses) / len(total_losses) if total_losses else float('inf')
        avg_cls  = sum(cls_losses)   / len(cls_losses)   if cls_losses   else 0.0
        avg_loc  = sum(loc_losses)   / len(loc_losses)   if loc_losses   else 0.0

        logger.info(
            f"Epoch {epoch+1} done in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} "
            f"(cls={avg_cls:.4f} loc={avg_loc:.4f})"
        )

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        trainer.save_checkpoint(epoch, 'latest.pth')
        trainer.save_checkpoint(epoch, 'best.pth', is_best=is_best)

        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            trainer.save_checkpoint(epoch, f'epoch_{epoch+1}.pth')

        torch.cuda.empty_cache()

        if trainer.interrupted:
            trainer.save_checkpoint(epoch, 'interrupted.pth')
            break

    total_time = time.time() - overall_start
    logger.info(
        f"\nTraining complete! Duration: "
        f"{str(datetime.timedelta(seconds=int(total_time)))}"
    )


if __name__ == '__main__':
    main()