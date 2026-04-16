# Copyright (c) SenseTime. All Rights Reserved.
"""
ModelBuilder: Defines the training and inference graph.
Fixes: Dynamic batch size, pure PyTorch tensor ops, device consistency.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, IOULoss
from pysot.models.backbone.newalexnet import AlexNet
from pysot.models.utile.utile import HiFT

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        # Initialize backbone and tracking head
        self.backbone = AlexNet()
        self.grader = HiFT(cfg)
        
        # Loss functions
        self.cls2loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IOULoss()

    def template(self, z):
        """Extract template features (used during inference initialization)."""
        with torch.no_grad():
            self.zf = self.backbone(z)

    def track(self, x):
        """Run tracking forward pass (used during inference)."""
        with torch.no_grad():
            xf = self.backbone(x)
            loc, cls1, cls2 = self.grader(xf, self.zf)
            return {'cls1': cls1, 'cls2': cls2, 'loc': loc}

    def log_softmax(self, cls):
        """Reshape classification output and apply log_softmax."""
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        return F.log_softmax(cls, dim=4)

    def get_center_targets(self, mapp):
        """
        Convert network output (delta predictions) to absolute bounding box coordinates.
        Fully tensor-based, device-agnostic, dynamic batch size.
        """
        b, c, h, w = mapp.size()
        device = mapp.device
        
        # Generate grid coordinates (0 to size-1)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        
        # Calculate physical coordinates in the 287x287 search image
        stride = cfg.ANCHOR.STRIDE  # 16
        offset = 63  # Centering offset for 287 size
        x_phys = x_grid * stride + offset - cfg.TRAIN.SEARCH_SIZE // 2
        y_phys = y_grid * stride + offset - cfg.TRAIN.SEARCH_SIZE // 2
        
        # Flatten grids for indexing
        x_phys = x_phys.view(-1)
        y_phys = y_phys.view(-1)
        xx = x_grid.view(-1).long()
        yy = y_grid.view(-1).long()

        # Inverse transform function (dcon)
        def dcon(x):
            x_clamped = torch.clamp(x, -0.99, 0.99)
            return (torch.log(1 + x_clamped) - torch.log(1 - x_clamped)) / 2.0

        # Decode width/height deltas
        deltas = dcon(mapp) * 143.0  # 143 is SEARCH_SIZE/2
        w_pred = deltas[:, 0, yy, xx] + deltas[:, 1, yy, xx]
        h_pred = deltas[:, 2, yy, xx] + deltas[:, 3, yy, xx]
        
        # Decode center coordinates
        cx = x_phys - deltas[:, 0, yy, xx] + w_pred / 2.0 + cfg.TRAIN.SEARCH_SIZE // 2
        cy = y_phys - deltas[:, 2, yy, xx] + h_pred / 2.0 + cfg.TRAIN.SEARCH_SIZE // 2

        # Convert to (x1, y1, x2, y2) format
        anchor = torch.zeros(b, h * w, 4, device=device)
        anchor[:, :, 0] = cx - w_pred / 2.0
        anchor[:, :, 1] = cy - h_pred / 2.0
        anchor[:, :, 2] = cx + w_pred / 2.0
        anchor[:, :, 3] = cy + h_pred / 2.0
        
        return anchor

    def forward(self, data):
        """
        Training forward pass.
        Expects data dict with keys: template, search, bbox, label_cls1, labelxff, labelcls2, weightxff
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        bbox = data['bbox'].cuda()
        label_cls1 = data['label_cls1'].cuda()
        label_xff = data['labelxff'].cuda()
        label_cls2 = data['labelcls2'].cuda()
        weight_xff = data['weightxff'].cuda()

        # Feature extraction
        zf = self.backbone(template)
        xf = self.backbone(search)
        
        # Tracking head
        loc, cls1, cls2 = self.grader(xf, zf)
        
        # Classification loss
        cls1_log = self.log_softmax(cls1)
        cls_loss1 = select_cross_entropy_loss(cls1_log, label_cls1)
        cls_loss2 = self.cls2loss(cls2, label_cls2)
        cls_loss = cfg.TRAIN.w4 * cls_loss1 + cfg.TRAIN.w5 * cls_loss2
        
        # Localization loss (IoU)
        pred_bbox = self.get_center_targets(loc)
        target_bbox = self.get_center_targets(label_xff)
        loc_loss = cfg.TRAIN.w3 * self.iou_loss(pred_bbox, target_bbox, weight_xff)
        
        # Total loss
        total_loss = cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CLS_WEIGHT * cls_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'loc_loss': loc_loss
        }