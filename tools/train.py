#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HiFT Fine-Tuning Script with Automatic 'Best Model' Saving
"""
# Add this after your imports to fix path issues
import os
import sys

# Get the absolute path to the project root (parent of 'tools')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Update default config path to be absolute
DEFAULT_CFG = os.path.join(PROJECT_ROOT, 'configs', 'hiFT_finetune.yaml')


import time
import datetime
import signal
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, add_file_handler
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.distributed import dist_init


# ROBUST LOGGER SETUP (fixes Spyder 'NoneType' issue)
def get_logger(name='global', level=logging.INFO):
    """Get or create a logger that works in any environment"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Logger not yet configured
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger('global', logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='HiFT Fine-tuning')
    parser.add_argument('--cfg', type=str, default=DEFAULT_CFG, help='config file')
    parser.add_argument('--resume', type=str, default=None, help='path to specific checkpoint to resume from')
    return parser.parse_args()

class SafeTrainer:
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.save_dir = cfg.TRAIN.SNAPSHOT_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        self.interrupted = False

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.warning("\n Interrupt received! Saving current state before exit...")
        self.interrupted = True

    def save_checkpoint(self, epoch, filename, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'cfg': self.cfg
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(state, path)
        logger.info(f" Saved checkpoint: {path}")
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(state, best_path)
            logger.info(f" New BEST model saved to: {best_path}")

def main():
    args = parse_args()
    cfg.merge_from_file(args.cfg)
    
    # Setup logging directory
    os.makedirs(cfg.TRAIN.LOG_DIR, exist_ok=True)
    try:
        add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'train.log'))
    except:
        pass  # Skip if handler already exists
    
    logger.info(f"Config loaded from {args.cfg}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Init distributed
    rank, world_size = dist_init()
    
    # 1. Build Model
    logger.info("Building Model...")
    model = ModelBuilder()
    
    # 2. Setup Optimizer
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.grader.parameters())
    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR},
        {'params': head_params, 'lr': cfg.TRAIN.BASE_LR}
    ], momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    
    # 3. Resume Logic
    start_epoch = 0
    resume_path = args.resume or os.path.join(cfg.TRAIN.SNAPSHOT_DIR, 'latest.pth')
    
    if os.path.exists(resume_path):
        logger.info(f" Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f" Resumed successfully. Starting from Epoch {start_epoch}")
    elif cfg.BACKBONE.PRETRAINED and os.path.exists(cfg.BACKBONE.PRETRAINED):
        load_pretrain(model.backbone, cfg.BACKBONE.PRETRAINED)
        logger.info(f"Loaded backbone weights from {cfg.BACKBONE.PRETRAINED}")
        
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH, last_epoch=start_epoch - 1)
    
    # 4. Load Dataset
    logger.info("Loading Dataset...")
    dataset = TrkDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset ready: {len(dataset)} samples | Batch size: {cfg.TRAIN.BATCH_SIZE}")
    
    trainer = SafeTrainer(model, optimizer, cfg)
    total_epochs = cfg.TRAIN.EPOCH
    best_loss = float('inf')
    
    logger.info(f" Starting fine-tuning: Epoch {start_epoch} to {total_epochs}")
    overall_start = time.time()

    for epoch in range(start_epoch, total_epochs):
        if trainer.interrupted:
            break
            
        model.train()
        epoch_start = time.time()
        epoch_losses = []
        
        logger.info(f"\n{'='*50}\n Epoch [{epoch+1}/{total_epochs}]")
        
        for i, data in enumerate(dataloader):
            if trainer.interrupted:
                break
                
            outputs = model(data)
            loss = outputs['total_loss']
            
            optimizer.zero_grad()
            loss.backward()
            
            if cfg.TRAIN.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                
            optimizer.step()
            epoch_losses.append(loss.item())
            
            if i % cfg.TRAIN.PRINT_FREQ == 0:
                lr = lr_scheduler.get_cur_lr()
                now = time.time()
                elapsed = now - overall_start
                iter_time = (now - epoch_start) / (i + 1)
                remaining_iters = (total_epochs - epoch - 1) * len(dataloader) + (len(dataloader) - i - 1)
                eta = iter_time * remaining_iters
                
                logger.info(
                    f" Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))} | "
                    f"ETA: {str(datetime.timedelta(seconds=int(eta)))} | "
                    f"Iter [{i}/{len(dataloader)}] | "
                    f"Loss: {loss.item():.4f} | LR: {lr:.6f}"
                )
                
        epoch_time = time.time() - epoch_start
        lr_scheduler.step()
        
        # AUTO-SAVE "BEST" MODEL
        current_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        if current_epoch_loss < best_loss:
            best_loss = current_epoch_loss
            trainer.save_checkpoint(epoch, 'best.pth', is_best=True)
            logger.info(f" Best model updated! New Loss: {best_loss:.4f}")
        
        # Save latest checkpoint
        trainer.save_checkpoint(epoch, 'latest.pth')
        
        # Save milestone checkpoints
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            trainer.save_checkpoint(epoch, f'epoch_{epoch+1}.pth')
            
        torch.cuda.empty_cache()
        logger.info(f" Epoch {epoch+1} completed in {epoch_time:.2f}s | Avg Loss: {current_epoch_loss:.4f}")
        
        if trainer.interrupted:
            trainer.save_checkpoint(epoch, 'latest.pth')
            break

    total_time = time.time() - overall_start
    logger.info(f"\n Training finished! Total duration: {str(datetime.timedelta(seconds=int(total_time)))}")

if __name__ == '__main__':
    main()