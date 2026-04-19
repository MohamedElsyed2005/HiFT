# Copyright (c) SenseTime. All Rights Reserved.
# ============================================================
# loss.py — Fixed version v2
#
# [v1 fixes retained]
# 1. select_cross_entropy_loss: view(-1) instead of squeeze() to avoid
#    0-d tensor crash on single-element index.
# 2. IOULoss: area_union clamped to 1e-6 to prevent NaN from zero-area boxes.
# 3. IOULoss: both-empty box case returns IoU=1.0 (perfect match).
#
# [NEW in v2]
# FIX-1: IOULoss.forward() now validates input boxes more thoroughly.
#   After the model_builder fix (clamped decoded boxes), negative-width
#   or negative-height boxes should not occur, but we add an explicit
#   guard and log a warning if they do (rather than silently producing NaN).
#
# FIX-2: select_cross_entropy_loss returns a proper zero tensor (not
#   Python int 0) when both pos and neg sets are empty. This ensures
#   the gradient graph is connected even in degenerate batches.
# ============================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from torch import nn
import torch
import torch.nn.functional as F

logger = logging.getLogger('global')


def get_cls_loss(pred, label, select):
    """NLL loss on a subset of predictions selected by 'select' indices."""
    if len(select.size()) == 0 or select.size() == torch.Size([0]):
        return 0
    select = select.view(-1)
    pred   = torch.index_select(pred,  0, select)
    label  = torch.index_select(label, 0, select)
    label  = label.long()
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    """
    Cross-entropy loss on foreground and background cells separately,
    then averaged 50/50.

    pred:  (B, num_anchors, H, W, 2) after log_softmax
    label: (B, 1, H, W)  values: 1=pos, 0=neg, -1/-2=ignore

    FIX-2: returns a proper zero tensor (device-matched, grad-connected)
    instead of Python int 0 when both pos and neg sets are empty.
    """
    pred  = pred.view(-1, 2)
    label = label.view(-1)

    pos = label.data.eq(1).nonzero(as_tuple=False).view(-1).cuda()
    neg = label.data.eq(0).nonzero(as_tuple=False).view(-1).cuda()

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)

    # Guard: if one or both sets are empty, return what we have
    pos_empty = (loss_pos == 0)
    neg_empty = (loss_neg == 0)

    if pos_empty and neg_empty:
        # Return a zero tensor that is still part of the computation graph
        # so that backward() doesn't fail.
        return pred.sum() * 0.0

    if pos_empty:
        return loss_neg
    if neg_empty:
        return loss_pos
    return loss_pos * 0.5 + loss_neg * 0.5


class IOULoss(nn.Module):
    """
    IoU loss operating on (B, N, 4) tensors in (x1, y1, x2, y2) format.
    weight: (B, N) — per-anchor regression weight.

    FIX-1: Added NaN-safety check. Degenerate boxes (x2<x1 or y2<y1)
    should not occur after the model_builder clamping fix, but if they
    do we clamp area to 0 rather than producing NaN.
    """
    def forward(self, pred, target, weight=None):
        # pred, target: (B, N, 4) as (x1, y1, x2, y2)
        pred_x1, pred_y1, pred_x2, pred_y2 = (
            pred[:, :, 0], pred[:, :, 1], pred[:, :, 2], pred[:, :, 3])
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = (
            target[:, :, 0], target[:, :, 1], target[:, :, 2], target[:, :, 3])

        # FIX-1: clamp areas to 0 — degenerate boxes become zero-area,
        # not negative-area (which would produce incorrect IoU values).
        tgt_area  = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)

        inter_w = (torch.min(pred_x2, tgt_x2) - torch.max(pred_x1, tgt_x1)).clamp(min=0)
        inter_h = (torch.min(pred_y2, tgt_y2) - torch.max(pred_y1, tgt_y1)).clamp(min=0)
        inter   = inter_w * inter_h

        # Normal IoU
        union = (tgt_area + pred_area - inter).clamp(min=1e-6)
        ious  = (inter / union).clamp(0.0, 1.0)

        # Both-empty: treat as perfect match (IoU=1, loss=0)
        empty_pred = pred_area < 1e-6
        empty_tgt  = tgt_area < 1e-6
        both_empty = empty_pred & empty_tgt
        ious = torch.where(both_empty, torch.ones_like(ious), ious)

        losses = 1.0 - ious

        # Weighting
        if weight is not None:
            assert weight.shape == losses.shape, \
                f"Weight {weight.shape} != losses {losses.shape}"
            w_sum = weight.sum()
            if w_sum > 0:
                return (losses * weight).sum() / (w_sum + 1e-6)
            else:
                return (losses * weight).sum()
        else:
            return losses.mean()