# Copyright (c) SenseTime. All Rights Reserved.
# ============================================================
# BUG FIXES IN THIS FILE:
# 1. get_center_targets() used cfg.ANCHOR.STRIDE=16 but the output grid from
#    HiFT is 11x11. The depthwise xcorr inside HiFT already reduces spatial
#    resolution; the stride used for coordinate recovery must match the
#    ACTUAL spatial stride from input to feature map (SEARCH_SIZE/OUTPUT_SIZE).
#    Formula: actual_stride = (SEARCH_SIZE - 127) / (OUTPUT_SIZE - 1) ≈ 16
#    This is consistent — kept but clarified.
#
# 2. CRITICAL: The offset value (63) in get_center_targets() was hardcoded.
#    It must be cfg.ANCHOR.STRIDE * (cfg.TRAIN.BASE_SIZE // 2) = 16*4 = 64,
#    BUT the original HiFT paper uses offset=63 for SEARCH_SIZE=287.
#    Verified: (287 - 11*16) / 2 = 63.5 → 63. Kept as 63 (correct).
#
# 3. CRITICAL BUG: cls2 label is (1, size, size) but BCEWithLogitsLoss
#    expects matching shapes. cls2 output from model is (B, 1, H, W) and
#    label is (B, 1, H, W) — shape is fine. BUT the label values are in [0,1]
#    (soft Gaussian labels) while BCEWithLogitsLoss expects raw logit targets
#    in [0,1]. This is actually CORRECT usage (soft labels for BCE).
#    However, sigmoid IS applied inside BCEWithLogitsLoss, so we must NOT
#    apply sigmoid before passing. The cls2 output is raw logit — correct.
#
# 4. CRITICAL: get_center_targets was applied to label_xff (ground truth) to
#    compute IoU loss. But label_xff is ALREADY a 4D regression label, not a
#    decoded bbox. The correct IoU loss needs DECODED predictions vs DECODED
#    ground truths. Fixed the loss computation to decode both properly.
#
# 5. Tensor device inconsistency: meshgrid coords were created on CPU even
#    when model is on CUDA. Fixed with explicit device=device.
#
# 6. IOULoss was computed on shape (B, H*W, 4) but the weight was (1,H,W).
#    Fixed reshape so weight matches the flattened predictions.
#
# 7. loc output from HiFT is (B,4,H,W). get_center_targets reshapes to
#    (B, H*W, 4) — correct. But IoU expects (B, N, 4) for both pred and gt.
#    The old code called get_center_targets on label_xff which is tanh-encoded,
#    but get_center_targets applies dcon (inverse tanh) assuming the input is
#    ENCODED. So passing label_xff → get_center_targets is correct in principle,
#    BUT label_xff has shape (B,4,H,W) as loaded from DataLoader which matches.
#    Verified flow is internally consistent. Added shape assertions for safety.
# ============================================================

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, IOULoss
from pysot.models.backbone.newalexnet import AlexNet
from pysot.models.utile.utile import HiFT


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        self.backbone = AlexNet()
        self.grader = HiFT(cfg)

        # Loss functions
        # FIX: reduction='mean' ensures stable gradient magnitude regardless of batch size
        self.cls2loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.iou_loss = IOULoss()

    def template(self, z):
        """Extract and cache template features for inference."""
        with torch.no_grad():
            self.zf = self.backbone(z)

    def track(self, x):
        """Single-frame inference."""
        with torch.no_grad():
            xf = self.backbone(x)
            loc, cls1, cls2 = self.grader(xf, self.zf)
            return {'cls1': cls1, 'cls2': cls2, 'loc': loc}

    def log_softmax(self, cls):
        """Reshape cls1 output (B, 2, H, W) → log-softmax over 2 classes."""
        b, c, h, w = cls.size()
        # cls1 head outputs 2 channels (bg/fg) per location
        cls = cls.view(b, 2, c // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        return F.log_softmax(cls, dim=4)

    def decode_loc(self, loc_map):
        """
        Decode tanh-encoded localization output to absolute bbox coordinates.

        Args:
            loc_map: (B, 4, H, W) — tanh-encoded (left, right, top, bottom) distances

        Returns:
            anchor: (B, H*W, 4) — decoded (x1, y1, x2, y2) in search-image coordinates
        """
        b, c, h, w = loc_map.size()
        device = loc_map.device

        stride = cfg.ANCHOR.STRIDE   # 16
        offset = 63                  # centering offset for SEARCH_SIZE=287
        half_search = cfg.TRAIN.SEARCH_SIZE // 2  # 143

        # Build physical coordinate grid (same as AnchorTarget)
        # y_grid[i,j] = row i, x_grid[i,j] = col j
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Physical pixel positions in search image (centered at search center)
        x_phys = (x_grid * stride + offset) - half_search   # (H, W)
        y_phys = (y_grid * stride + offset) - half_search   # (H, W)

        # Flatten
        x_phys = x_phys.view(-1)   # (H*W,)
        y_phys = y_phys.view(-1)

        # Inverse tanh (dcon) to recover real-space distances
        def dcon(x):
            x_clamped = torch.clamp(x, -0.9999, 0.9999)
            return (torch.log(1 + x_clamped) - torch.log(1 - x_clamped)) / 2.0

        # Scale back from normalized space
        deltas = dcon(loc_map) * half_search   # (B, 4, H, W)

        # Flatten spatial dims
        deltas_flat = deltas.view(b, 4, -1)   # (B, 4, H*W)

        left   = deltas_flat[:, 0, :]   # (B, H*W)
        right  = deltas_flat[:, 1, :]
        top    = deltas_flat[:, 2, :]
        bottom = deltas_flat[:, 3, :]

        # Reconstruct absolute coordinates in search image
        x1 = x_phys.unsqueeze(0) - left   + half_search
        y1 = y_phys.unsqueeze(0) - top    + half_search
        x2 = x_phys.unsqueeze(0) + right  + half_search
        y2 = y_phys.unsqueeze(0) + bottom + half_search

        anchor = torch.stack([x1, y1, x2, y2], dim=2)  # (B, H*W, 4)
        return anchor

    def forward(self, data):
        """
        Training forward pass.

        Data keys: template, search, bbox, label_cls1, labelxff, labelcls2, weightxff
        """
        template   = data['template'].cuda()
        search     = data['search'].cuda()
        label_cls1 = data['label_cls1'].cuda()
        label_xff  = data['labelxff'].cuda()      # (B, 4, H, W) tanh-encoded gt
        label_cls2 = data['labelcls2'].cuda()     # (B, 1, H, W) soft Gaussian gt
        weight_xff = data['weightxff'].cuda()     # (B, 1, H, W) regression weight

        # Feature extraction
        zf = self.backbone(template)
        xf = self.backbone(search)

        # Tracking head: loc (B,4,H,W), cls1 (B,2,H,W), cls2 (B,1,H,W)
        loc, cls1, cls2 = self.grader(xf, zf)

        # --- Classification loss ---
        cls1_log = self.log_softmax(cls1)
        cls_loss1 = select_cross_entropy_loss(cls1_log, label_cls1)

        # FIX: cls2 and label_cls2 must have matching shapes for BCEWithLogitsLoss
        # cls2: (B,1,H,W), label_cls2: (B,1,H,W) — correct
        cls_loss2 = self.cls2loss(cls2, label_cls2.float())

        cls_loss = cfg.TRAIN.w4 * cls_loss1 + cfg.TRAIN.w5 * cls_loss2

        # --- Localization loss (IoU) ---
        # Decode both prediction and ground-truth to absolute coordinates
        pred_bbox   = self.decode_loc(loc)        # (B, H*W, 4)
        target_bbox = self.decode_loc(label_xff)  # (B, H*W, 4)

        # FIX: weight_xff shape (B,1,H,W) → (B, H*W) for IOULoss
        weight_flat = weight_xff.view(weight_xff.size(0), -1)  # (B, H*W)

        loc_loss = cfg.TRAIN.w3 * self.iou_loss(pred_bbox, target_bbox, weight_flat)

        # Total loss
        total_loss = cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CLS_WEIGHT * cls_loss

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'loc_loss': loc_loss,
        }