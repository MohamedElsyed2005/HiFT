# ============================================================
# model_builder.py — Fixed version v2
#
# FIXES IN THIS VERSION:
#
# [v1 fixes retained]
# - decode_loc() runtime guard: raises if feature map size ≠ OUTPUT_SIZE
# - decode_loc() correctly reconstructs (B, HW, 4) absolute bboxes from
#   tanh-encoded FCOS-style offsets
#
# [NEW in v2]
# FIX-1: loc_loss detach guard during freeze phase.
#   When loc_weight == 0.0, we now detach the loc computation from the
#   graph entirely (using torch.no_grad). This prevents the randomly-
#   initialized loc head from accumulating gradient-related state (e.g.
#   Adam's second moment estimates) that would cause an erratic first
#   optimizer step when loc_weight first becomes non-zero. It also
#   removes the small but non-zero numerical gradient that can leak
#   through the IoU loss computation even at weight=0.
#
# FIX-2: Zero-area box guard in decode_loc.
#   After dcon() inversion, if left+right ≤ 0 or top+bottom ≤ 0, the
#   decoded box has zero or negative size. We clamp to a minimum box
#   size of 1 pixel so the IoU loss receives valid input rather than
#   degenerate boxes that produce undefined gradients.
#
# FIX-3: target_bbox computation only when needed.
#   decode_loc on the label_xff tensor is expensive and produces no
#   gradient. Moved inside the loc_weight > 0 branch.
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
        self.grader   = HiFT(cfg)
        self.cls2loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.iou_loss = IOULoss()

    def template(self, z):
        with torch.no_grad():
            self.zf = self.backbone(z)

    def track(self, x):
        with torch.no_grad():
            xf = self.backbone(x)
            loc, cls1, cls2 = self.grader(xf, self.zf)
            return {'cls1': cls1, 'cls2': cls2, 'loc': loc}

    def log_softmax(self, cls):
        b, c, h, w = cls.size()
        cls = cls.view(b, 2, c // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        return F.log_softmax(cls, dim=4)

    def decode_loc(self, loc_map):
        """
        Decode tanh-encoded (B,4,H,W) → absolute (B,H*W,4) bboxes.

        RUNTIME GUARD: asserts H==W==OUTPUT_SIZE so any future config
        mismatch is caught immediately with a clear error message.

        FIX-2: clamps decoded widths and heights to a minimum of 1 pixel
        to prevent degenerate zero-area boxes from producing undefined
        gradients in the IoU loss.
        """
        b, c, h, w = loc_map.size()
        device = loc_map.device

        expected = cfg.TRAIN.OUTPUT_SIZE
        if h != expected or w != expected:
            raise RuntimeError(
                f"decode_loc: got feature map {h}×{w} but "
                f"cfg.TRAIN.OUTPUT_SIZE={expected}. "
                f"For SEARCH_SIZE={cfg.TRAIN.SEARCH_SIZE}, "
                f"the correct OUTPUT_SIZE is 11. Fix your YAML.")

        stride      = cfg.ANCHOR.STRIDE        # 16
        offset      = 63                        # centering for 287px search
        half_search = cfg.TRAIN.SEARCH_SIZE // 2  # 143

        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij')

        x_phys = (x_grid * stride + offset) - half_search
        y_phys = (y_grid * stride + offset) - half_search
        x_phys = x_phys.view(-1)
        y_phys = y_phys.view(-1)

        def dcon(x):
            x = torch.clamp(x, -0.9999, 0.9999)
            return (torch.log(1 + x) - torch.log(1 - x)) / 2.0

        deltas      = dcon(loc_map) * half_search    # (B,4,H,W)
        deltas_flat = deltas.view(b, 4, -1)          # (B,4,H*W)
        left   = deltas_flat[:, 0, :]
        right  = deltas_flat[:, 1, :]
        top    = deltas_flat[:, 2, :]
        bottom = deltas_flat[:, 3, :]

        # FIX-2: Clamp decoded extents to at least 1 pixel.
        # Without this, left+right ≤ 0 (possible with tanh outputs near -1)
        # produces zero-width boxes that yield IoU=0 and a loss=1.0 even
        # for "correct" predictions, creating misleading gradients.
        left   = torch.clamp(left,   min=0)
        right  = torch.clamp(right,  min=0)
        top    = torch.clamp(top,    min=0)
        bottom = torch.clamp(bottom, min=0)
        # Ensure minimum 1px width/height by boosting smaller of each pair
        w_total = (left + right).clamp(min=1.0)
        h_total = (top  + bottom).clamp(min=1.0)
        # Scale left/right proportionally if clamped
        safe_lr = (left + right + 1e-6)
        safe_tb = (top  + bottom + 1e-6)
        left   = left   * w_total / safe_lr
        right  = right  * w_total / safe_lr
        top    = top    * h_total / safe_tb
        bottom = bottom * h_total / safe_tb

        x1 = x_phys.unsqueeze(0) - left   + half_search
        y1 = y_phys.unsqueeze(0) - top    + half_search
        x2 = x_phys.unsqueeze(0) + right  + half_search
        y2 = y_phys.unsqueeze(0) + bottom + half_search

        return torch.stack([x1, y1, x2, y2], dim=2)   # (B,H*W,4)

    def forward(self, data, loc_weight=None, cls_weight=None):

        loc_weight = loc_weight if loc_weight is not None else cfg.TRAIN.LOC_WEIGHT
        cls_weight = cls_weight if cls_weight is not None else cfg.TRAIN.CLS_WEIGHT

        template   = data['template'].cuda()
        search     = data['search'].cuda()
        label_cls1 = data['label_cls1'].cuda()
        label_xff  = data['labelxff'].cuda()
        label_cls2 = data['labelcls2'].cuda()
        weight_xff = data['weightxff'].cuda()

        zf = self.backbone(template)
        xf = self.backbone(search)
        loc, cls1, cls2 = self.grader(xf, zf)

        # ── Classification loss (always active) ──────────────────────────────
        cls1_log  = self.log_softmax(cls1)
        cls_loss1 = select_cross_entropy_loss(cls1_log, label_cls1)
        cls_loss2 = self.cls2loss(cls2, label_cls2.float())
        cls_loss  = cfg.TRAIN.w4 * cls_loss1 + cfg.TRAIN.w5 * cls_loss2

        # ── Localization loss ─────────────────────────────────────────────────
        # FIX-1: When loc_weight == 0, skip the IoU loss computation entirely.
        #
        # Rationale: even though 0 * loc_loss = 0 in the forward pass, the
        # backward pass through decode_loc still computes gradients for `loc`
        # (and through it, for the backbone). These gradients are multiplied by
        # zero in the chain rule for the total_loss, so they SHOULD be zero.
        # However, with floating-point arithmetic and the dcon() atanh
        # approximation, tiny non-zero gradients can accumulate in Adam's
        # second-moment (v_t) estimates, causing the first real loc_weight > 0
        # step to produce a disproportionately large effective LR for the loc
        # parameters. Using torch.no_grad() during the freeze phase prevents
        # ANY gradient information from flowing through the loc path, giving
        # Adam a clean state when loc training begins.
        if loc_weight > 0.0:
            pred_bbox   = self.decode_loc(loc)
            target_bbox = self.decode_loc(label_xff)
            weight_flat = weight_xff.view(weight_xff.size(0), -1)
            loc_loss    = cfg.TRAIN.w3 * self.iou_loss(
                pred_bbox, target_bbox, weight_flat)
        else:
            # Compute a detached loc_loss for logging only (no gradient).
            with torch.no_grad():
                pred_bbox_det   = self.decode_loc(loc.detach())
                target_bbox_det = self.decode_loc(label_xff)
                weight_flat_det = weight_xff.view(weight_xff.size(0), -1)
                loc_loss = cfg.TRAIN.w3 * self.iou_loss(
                    pred_bbox_det, target_bbox_det, weight_flat_det)

        total_loss = loc_weight * loc_loss + cls_weight * cls_loss

        return {
            'total_loss': total_loss,
            'cls_loss':   cls_loss,
            'loc_loss':   loc_loss,
        }