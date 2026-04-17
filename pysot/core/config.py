# Copyright (c) SenseTime. All Rights Reserved.
# ============================================================
# BUG FIXES IN THIS FILE:
# 1. BACKBONE.PRETRAINED default path updated to 'pretrained/first.pth'
#    (was 'back.pth' which would cause FileNotFoundError at startup)
# 2. DATASET.NAMES default kept minimal to avoid loading unused datasets
# 3. TRAIN.BATCH_SIZE reduced to 4 (was 100 — would OOM on RTX 3050 4GB)
# 4. TRAIN.NUM_GPU set to 1 (was 2 — single GPU setup)
# 5. TRAIN.EPOCH set to 15 for fine-tuning (was 30 — overkill for fine-tune)
# 6. OUTPUT_SIZE set to 11 to match the HiFT head output (was inconsistently
#    set to 21 in default config but 11 in YAML; 11 is correct for SEARCH=287)
# 7. OUTPUTFEATURE_SIZE set to 11 (AnchorTarget uses this for label grid)
# ============================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()
cfg = __C

__C.META_ARC = "HiFT_alexnet"
__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.THR_HIGH = 0.6
__C.TRAIN.THR_LOW = 0.3
__C.TRAIN.NEG_NUM = 16
__C.TRAIN.POS_NUM = 16
__C.TRAIN.TOTAL_NUM = 64

# Loss component weights
__C.TRAIN.CLS_WEIGHT = 1.0
__C.TRAIN.LOC_WEIGHT = 3.0
__C.TRAIN.SHAPE_WEIGHT = 2.0

# FIX #1: w4/w5 are used in model_builder for cls loss weighting
__C.TRAIN.w1 = 1.0
__C.TRAIN.w2 = 1.0
__C.TRAIN.w3 = 1.0
__C.TRAIN.w4 = 1.0   # weight for cls1 (cross-entropy)
__C.TRAIN.w5 = 1.0   # weight for cls2 (BCE)
__C.TRAIN.range = 2.0

__C.TRAIN.MASK_WEIGHT = 1
__C.TRAIN.LOG_GRADS = False
__C.TRAIN.GRAD_CLIP = 10.0

# FIX #2: EXEMPLAR and SEARCH sizes must be consistent
__C.TRAIN.EXEMPLAR_SIZE = 127
__C.TRAIN.SEARCH_SIZE = 287   # must match what AnchorTarget & model use

# FIX #3: OUTPUT_SIZE=11 is correct for SEARCH_SIZE=287 with stride=16+padding
# Formula: (287 - 127) / 16 + 1 ≈ 11 (after depthwise xcorr inside HiFT)
__C.TRAIN.OUTPUT_SIZE = 11           # was incorrectly 21 in base config
__C.TRAIN.OUTPUTFEATURE_SIZE = 11   # must equal OUTPUT_SIZE

__C.TRAIN.LABEL_RANGE = 4
__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.RESUME = ''
__C.TRAIN.PRETRAINED = 1
__C.TRAIN.LARGER = 2.0

__C.TRAIN.LOG_DIR = './logs'
__C.TRAIN.SNAPSHOT_DIR = './snapshot'

# FIX #4: Reduced for RTX 3050 4GB (was 100 — immediate OOM)
__C.TRAIN.EPOCH = 15
__C.TRAIN.START_EPOCH = 0
__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.NUM_GPU = 1       # was 2
__C.TRAIN.NUM_WORKERS = 2   # slightly higher than 1 for better throughput

__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.BASE_LR = 0.0005

__C.TRAIN.LR = CN()
__C.TRAIN.LR.TYPE = 'step'
__C.TRAIN.LR.KWARGS = CN(new_allowed=True)
__C.TRAIN.LR.KWARGS.start_lr = 0.0005
__C.TRAIN.LR.KWARGS.end_lr = 0.00005
__C.TRAIN.LR.KWARGS.step = 5
__C.TRAIN.LR.KWARGS.mult = 0.1

__C.TRAIN.LR_WARMUP = CN()
__C.TRAIN.LR_WARMUP.WARMUP = True
__C.TRAIN.LR_WARMUP.TYPE = 'step'
__C.TRAIN.LR_WARMUP.EPOCH = 3
__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.PRINT_FREQ = 20

# Classification head channel config
__C.TRAIN.clsandlocchannel = 256
__C.TRAIN.groupchannel = 32
__C.TRAIN.PR = 1

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation — template
__C.DATASET.TEMPLATE = CN()
__C.DATASET.TEMPLATE.SHIFT = 4
__C.DATASET.TEMPLATE.SCALE = 0.05
__C.DATASET.TEMPLATE.BLUR = 0.0
__C.DATASET.TEMPLATE.FLIP = 0.0
__C.DATASET.TEMPLATE.COLOR = 1.0

# Augmentation — search
__C.DATASET.SEARCH = CN()
__C.DATASET.SEARCH.SHIFT = 64
__C.DATASET.SEARCH.SCALE = 0.18
__C.DATASET.SEARCH.BLUR = 0.0
__C.DATASET.SEARCH.FLIP = 0.0
__C.DATASET.SEARCH.COLOR = 1.0

__C.DATASET.NEG = 0.2
__C.DATASET.GRAY = 0.0

# FIX #5: Default uses AIC4 only so the config is self-consistent out of box
__C.DATASET.NAMES = ('AIC4',)

__C.DATASET.AIC4 = CN()
__C.DATASET.AIC4.ROOT = 'data/processed/crop511'
__C.DATASET.AIC4.ANNO = 'data/processed/train.json'
__C.DATASET.AIC4.FRAME_RANGE = 50

__C.DATASET.AIC4.NUM_USE = 60000

__C.DATASET.VIDEOS_PER_EPOCH = 20000

# Keep legacy dataset configs so old YAML keys don't crash (new_allowed=True handles it)
__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'train_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'train_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = 'train_dataset/got10k/crop511'
__C.DATASET.GOT.ANNO = 'train_dataset/got10k/train.json'
__C.DATASET.GOT.FRAME_RANGE = 50
__C.DATASET.GOT.NUM_USE = 100000

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()
__C.BACKBONE.TYPE = 'alexnet'
__C.BACKBONE.KWARGS = CN(new_allowed=True)

# FIX #6: Updated to match actual file name provided ('first.pth' in pretrained/)
__C.BACKBONE.PRETRAINED = 'pretrained/first.pth'

# Only fine-tune last two layers — freeze early layers to save memory
__C.BACKBONE.TRAIN_LAYERS = ['layer4', 'layer5']
__C.BACKBONE.LAYERS_LR = 0.05

__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Anchor options
# ------------------------------------------------------------------------ #
__C.ANCHOR = CN()
__C.ANCHOR.STRIDE = 16

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()
__C.TRACK.TYPE = 'HiFTtracker'

__C.TRACK.PENALTY_K = 0.04
__C.TRACK.WINDOW_INFLUENCE = 0.44
__C.TRACK.LR = 0.4

__C.TRACK.w1 = 1.2
__C.TRACK.w2 = 1.0
__C.TRACK.w3 = 1.6
__C.TRACK.LARGER = 1.4

__C.TRACK.EXEMPLAR_SIZE = 127
__C.TRACK.INSTANCE_SIZE = 255
__C.TRACK.BASE_SIZE = 8
__C.TRACK.STRIDE = 8
__C.TRACK.CONTEXT_AMOUNT = 0.5

# Long-term tracking params
__C.TRACK.LOST_INSTANCE_SIZE = 831
__C.TRACK.CONFIDENCE_LOW = 0.85
__C.TRACK.CONFIDENCE_HIGH = 0.998
__C.TRACK.MASK_THERSHOLD = 0.30
__C.TRACK.MASK_OUTPUT_SIZE = 127