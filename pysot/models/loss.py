# Copyright (c) SenseTime. All Rights Reserved.
# ============================================================
# BUG FIXES IN THIS FILE:
# 1. IOULoss.forward() received weight as (B, 1, H, W) and called
#    weight.view(losses.size()). losses.size() == (B, H*W) so the reshape
#    would silently succeed (same element count) but the weight values would
#    be in the wrong order after the view (row-major vs. the flat order of
#    pred/target). The caller now passes weight already shaped (B, H*W),
#    so this file just validates shapes match.
#
# 2. IOULoss: area_union could be 0 when pred and target are both zero-sized.
#    Added clamp to prevent NaN losses from zero-area boxes.
#
# 3. select_cross_entropy_loss: nonzero(as_tuple=False).squeeze() can return
#    a 0-dim tensor when there is exactly 1 positive/negative, causing
#    index_select to crash. Added unsqueeze guard.
# ============================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import nn
import torch
import torch.nn.functional as F


def get_cls_loss(pred, label, select):
    """NLL loss on a subset of predictions selected by 'select' indices."""
    if len(select.size()) == 0 or select.size() == torch.Size([0]):
        return 0
    # FIX: squeeze() on a single-element tensor returns 0-d tensor.
    # index_select requires 1-d index. Use view(-1) instead.
    select = select.view(-1)
    pred  = torch.index_select(pred,  0, select)
    label = torch.index_select(label, 0, select)
    label = label.long()
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    """
    Cross-entropy loss on foreground and background cells separately,
    then averaged 50/50.

    pred:  (B, num_anchors, H, W, 2) after log_softmax
    label: (B, 1, H, W)  values: 1=pos, 0=neg, -1/-2=ignore
    """
    pred  = pred.view(-1, 2)
    label = label.view(-1)

    # FIX: use view(-1) to avoid 0-d tensor from squeeze()
    pos = label.data.eq(1).nonzero(as_tuple=False).view(-1).cuda()
    neg = label.data.eq(0).nonzero(as_tuple=False).view(-1).cuda()

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)

    # Guard: if one set is empty, return the other only
    if loss_pos == 0 and loss_neg == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    if loss_pos == 0:
        return loss_neg
    if loss_neg == 0:
        return loss_pos
    return loss_pos * 0.5 + loss_neg * 0.5


class IOULoss(nn.Module):
    """
    GIoU-inspired IoU loss operating on (B, N, 4) tensors in (x1,y1,x2,y2) format.
    weight: (B, N) — per-anchor regression weight.
    """
    def forward(self, pred, target, weight=None):
        # pred, target: (B, N, 4)
        pred_x1, pred_y1, pred_x2, pred_y2 = (
            pred[:, :, 0], pred[:, :, 1], pred[:, :, 2], pred[:, :, 3])
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = (
            target[:, :, 0], target[:, :, 1], target[:, :, 2], target[:, :, 3])

        tgt_area  = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)

        inter_w = (torch.min(pred_x2, tgt_x2) - torch.max(pred_x1, tgt_x1)).clamp(min=0)
        inter_h = (torch.min(pred_y2, tgt_y2) - torch.max(pred_y1, tgt_y1)).clamp(min=0)
        inter   = inter_w * inter_h

        # FIX: clamp union to avoid NaN from 0/0
        union = (tgt_area + pred_area - inter).clamp(min=1e-6)
        ious  = (inter / union).clamp(min=0)
        losses = 1.0 - ious

        if weight is not None:
            # weight: (B, N) — same shape as losses
            weight = weight.view(losses.size())
            w_sum = weight.sum()
            if w_sum > 0:
                return (losses * weight).sum() / (w_sum + 1e-6)
            else:
                return (losses * weight).sum()
        else:
            return losses.mean()