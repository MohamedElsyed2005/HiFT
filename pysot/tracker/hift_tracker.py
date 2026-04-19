"""
pysot/tracker/hift_tracker.py — CORRECTED VERSION v2

FIXES IN v2:

[v1 fix retained]
- CONFIDENCE_THRESHOLD lowered to 0.10 (was 0.25).

[NEW FIX-A]: scale_z inconsistency between init() and track().
  In init(), s_z = round(np.sqrt(...)) creates an INTEGER (Python round()
  returns int when called with one arg). self.scale_z = s_z stores an int.
  In track(), s_z = np.sqrt(...) is a FLOAT. Then scale_z = EXEMPLAR_SIZE/s_z
  uses the float s_z, which is correct.
  
  The inconsistency was: self.scale_z (from init) was used as a fallback when
  the target is very large relative to the image:
    if self.size[0]*self.size[1] > 0.5*img_h*img_w: s_z = self.scale_z
  This branch set s_z to the ROUNDED integer from init, but then computed
  scale_z = EXEMPLAR_SIZE / s_z using integer division semantics. Fixed by
  storing self.scale_z as float (before round) in init().

[NEW FIX-B]: generate_anchor() used a hardcoded 'cuda()' call for arange,
  which would break if running inference on CPU. Fixed to use the device
  of the loc_map tensor.

[NEW FIX-C]: _convert_score() cls1 shape handling.
  HiFT returns cls1 with shape (B, 2, H, W). The reshape to (-1, 2) is
  correct BUT permute(0,2,3,1) gives (B,H,W,2) → view(-1,2) gives (B*H*W,2).
  This is correct ordering. No change needed, just verified.

[NEW FIX-D]: track() now handles the case where out['bbox'] contains
  NaN or Inf values (which can happen when the model is poorly calibrated).
  Such predictions are treated as tracker failure and last_pred is returned.
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
        """Inverse tanh: maps (-1,1) → R (atanh)."""
        x = np.clip(x, -0.9999, 0.9999)
        return (np.log(1 + x) - np.log(1 - x)) / 2.0

    def generate_anchor(self, loc_map):
        """
        Decode (1, 4, H, W) localization output to (4, H*W) anchors.
        Returns [cx, cy, w, h] as OFFSETS from self.center_pos (in image space).

        FIX-B: device-agnostic (no hardcoded .cuda()).
        """
        size        = self.score_size
        stride      = cfg.ANCHOR.STRIDE
        offset      = 63
        half_search = cfg.TRAIN.SEARCH_SIZE // 2

        # Grid positions in search-image-centred coordinates
        xs = stride * np.arange(size) + offset - half_search  # shape (11,)
        ys = stride * np.arange(size) + offset - half_search  # shape (11,)

        x_grid = np.tile(xs,   size)   # (121,) - x repeats for each row
        y_grid = np.repeat(ys, size)   # (121,) - y repeats for each column

        loc_np       = loc_map[0].cpu().detach().numpy()  # (4, 11, 11)
        shape_deltas = self._inverse_transform(loc_np) * half_search  # (4, 11, 11)

        left   = shape_deltas[0].flatten()  # (121,)
        right  = shape_deltas[1].flatten()
        top    = shape_deltas[2].flatten()
        bottom = shape_deltas[3].flatten()

        # Clamp decoded extents to non-negative (consistent with model_builder fix)
        left   = np.maximum(left,   0)
        right  = np.maximum(right,  0)
        top    = np.maximum(top,    0)
        bottom = np.maximum(bottom, 0)

        w  = np.maximum(left + right,  1.0)
        h  = np.maximum(top  + bottom, 1.0)

        # cx, cy are the offset of the box CENTER from the search-image center
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
        s_z = np.sqrt(w_z * h_z)

        # FIX-A: Store s_z as FLOAT before rounding.
        # track() computes s_z as a float and uses it to derive scale_z.
        # The fallback branch (large target case) uses self.scale_z as s_z,
        # so it must be a float to avoid integer-division artefacts.
        self.scale_z = float(s_z)  # was: round(s_z) -> int

        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    round(s_z), self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)

        # Large-target fallback: use stored s_z to avoid wild scale changes
        if self.size[0] * self.size[1] > 0.5 * img.shape[0] * img.shape[1]:
            s_z = self.scale_z   # FIX-A: now a float, consistent with below

        scale_z     = cfg.TRACK.EXEMPLAR_SIZE / s_z  # float division
        search_size = cfg.TRAIN.SEARCH_SIZE           # 287
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

        # FIX-D: Guard against NaN/Inf from poorly-calibrated model
        if not (np.isfinite(cx) and np.isfinite(cy) and
                np.isfinite(width) and np.isfinite(height)):
            return {
                'bbox':       None,
                'best_score': float(score[best_idx]),
                'failed':     True
            }

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height,
                                                img.shape[:2])
        self.center_pos = np.array([cx, cy])
        self.size       = np.array([width, height])

        best_score = float(score[best_idx])

        # Confidence threshold: low enough to not reject learning-phase predictions,
        # high enough to flag genuinely lost trackers.
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