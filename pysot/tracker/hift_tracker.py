"""
pysot/tracker/hift_tracker.py — CORRECTED VERSION

FIXES:
  - CONFIDENCE_THRESHOLD lowered to 0.10 (was 0.25).
    During early training epochs the model produces low-confidence scores
    everywhere. The 0.25 threshold caused the tracker to declare failure on
    virtually every frame of every validation sequence, filling the val loop
    with IoU=0 entries and making AUC look catastrophically bad even when
    the model had learned something useful.
  - out['bbox'] return value is always a list or None (never a mixed type).
    Callers can safely check `out.get('failed', False)` or `out['bbox'] is None`.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker


class HiFTTracker(SiameseTracker):
    def __init__(self, model):
        super(HiFTTracker, self).__init__()
        self.model = model
        self.model.eval()

        self.score_size = cfg.TRAIN.OUTPUT_SIZE   # 11

        hanning     = np.hanning(self.score_size)
        self.window = np.outer(hanning, hanning).flatten()

    def _inverse_transform(self, x):
        """Inverse tanh: maps (-1,1) → R."""
        x = np.clip(x, -0.9999, 0.9999)
        return (np.log(1 + x) - np.log(1 - x)) / 2.0

    def generate_anchor(self, loc_map):
        """
        Decode (1, 4, H, W) localization output to (4, H*W) anchors.
        Returns [cx, cy, w, h] in search-image coordinates.
        """
        size        = self.score_size
        stride      = cfg.ANCHOR.STRIDE
        offset      = 63
        half_search = cfg.TRAIN.SEARCH_SIZE // 2

        xs = stride * np.arange(size) + offset - half_search
        ys = stride * np.arange(size) + offset - half_search

        x_grid = np.tile(xs,   size)
        y_grid = np.repeat(ys, size)

        loc_np     = loc_map[0].cpu().detach().numpy()
        shape_deltas = self._inverse_transform(loc_np) * half_search

        left   = shape_deltas[0].flatten()
        right  = shape_deltas[1].flatten()
        top    = shape_deltas[2].flatten()
        bottom = shape_deltas[3].flatten()

        w  = left + right
        h  = top  + bottom

        cx = x_grid - left + w / 2.0
        cy = y_grid - top  + h / 2.0

        anchor = np.zeros((4, size * size))
        anchor[0, :] = cx
        anchor[1, :] = cy
        anchor[2, :] = w
        anchor[3, :] = h
        return anchor

    def _convert_score(self, cls1):
        """(B, 2, H, W) → (H*W,) foreground probability via softmax."""
        score = cls1.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _convert_cls2_score(self, cls2):
        """(B, 1, H, W) raw logits → (H*W,) probability via sigmoid."""
        return torch.sigmoid(cls2).view(-1).cpu().detach().numpy()

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx     = max(0, min(cx,    boundary[1]))
        cy     = max(0, min(cy,    boundary[0]))
        width  = max(10, min(width,  boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        self.image      = img
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.scale_z = s_z

        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)

        if self.size[0] * self.size[1] > 0.5 * img.shape[0] * img.shape[1]:
            s_z = self.scale_z

        scale_z     = cfg.TRACK.EXEMPLAR_SIZE / s_z
        search_size = cfg.TRAIN.SEARCH_SIZE   # 287 — matches training input size
        s_x         = s_z * (search_size / cfg.TRACK.EXEMPLAR_SIZE)

        x_crop = self.get_subwindow(img, self.center_pos,
                                    search_size,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        pred_bbox = self.generate_anchor(outputs['loc'])

        score1 = self._convert_score(outputs['cls1'])
        score2 = self._convert_cls2_score(outputs['cls2'])

        weights = cfg.TRACK.SCORE_WEIGHTS
        score   = score1 * weights[0] + score2 * weights[1]

        def change(r):
            return np.maximum(r, 1.0 / (r + 1e-5))

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = change((self.size[0] / (self.size[1] + 1e-5)) /
                     (pred_bbox[2, :] / (pred_bbox[3, :] + 1e-5)))

        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore  = penalty * score
        pscore  = (pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE)
                   + self.window * cfg.TRACK.WINDOW_INFLUENCE)

        best_idx = np.argmax(pscore)
        bbox     = pred_bbox[:, best_idx] / scale_z
        lr       = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx     = bbox[0] + self.center_pos[0]
        cy     = bbox[1] + self.center_pos[1]
        width  = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height,
                                                img.shape[:2])
        self.center_pos = np.array([cx, cy])
        self.size       = np.array([width, height])

        best_score = float(score[best_idx])

        # FIX: Threshold lowered from 0.25 → 0.10.
        # During early training, the model's softmax/sigmoid scores are low
        # everywhere (the head hasn't learned to produce high-confidence responses
        # yet). A 0.25 threshold rejects nearly all predictions during early
        # epochs, filling the validation loop with IoU=0 entries from the
        # last_pred fallback. This made AUC appear to collapse even when the
        # model was learning. 0.10 is a conservative lower bound that still
        # rejects genuinely lost trackers while allowing learning-phase predictions.
        CONFIDENCE_THRESHOLD = 0.10

        if best_score < CONFIDENCE_THRESHOLD:
            return {
                'bbox':       None,
                'best_score': best_score,
                'failed':     True
            }

        return {
            'bbox':       [cx - width / 2, cy - height / 2, width, height],
            'best_score': best_score,
            'failed':     False
        }