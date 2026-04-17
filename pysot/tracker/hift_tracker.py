# Copyright (c) SenseTime. All Rights Reserved.
# ============================================================
# BUG FIXES IN THIS FILE:
# 1. CRITICAL: score_size used cfg.TRAIN.OUTPUT_SIZE (11) but the tracker runs
#    with INSTANCE_SIZE=255 (not SEARCH_SIZE=287 used in training).
#    At inference, the network processes a 255x255 crop, which yields a
#    different spatial output size than during training (287x287).
#    255 input → AlexNet → (255-127)/16 + 1 = 8.25 → but after xcorr in HiFT
#    the actual output is determined by the xcorr between xf and zf feature maps.
#    The feature map from AlexNet for 255 input: layer5 output = 6x6.
#    For 127 input: layer5 output = 6x6. xcorr → 1x1 per channel.
#    That doesn't match. Let's trace carefully:
#      - AlexNet layer1: stride 2, kernel 11 → 255→123, 127→59
#      - layer1 MaxPool: stride 2, kernel 3 → 123→61, 59→29
#      - layer2: kernel 5 → 61→57, 29→25
#      - layer2 MaxPool: stride 2, kernel 3 → 57→28, 25→12
#      - layer3: kernel 3 → 28→26, 12→10
#      - layer4: kernel 3 → 26→24, 10→8
#      - layer5: kernel 3 → 24→22, 8→6
#    xcorr depthwise: search_feat (22x22) conv with template_feat (6x6) → 17x17
#    Then HiFT conv1 stride=2 → 9x9. But training uses 287→ layer5=27,
#    xcorr(27,6)=22, conv1 stride=2 → 11x11.
#    So inference with 255 gives 9x9, training gives 11x11. THIS IS A MISMATCH.
#
#    FIX: Use SEARCH_SIZE=287 at inference too, matching training.
#    Set INSTANCE_SIZE=287 in TRACK config OR pass search size explicitly.
#    We fix by using cfg.TRAIN.SEARCH_SIZE (287) for the crop at inference.
#    NOTE: This also means score_size=11, consistent with training labels.
#
# 2. generate_anchor() hardcoded stride=cfg.ANCHOR.STRIDE (16) but that is
#    the feature-map-to-search-image stride, which we verified is correct
#    for 287 input. Updated offset to match training formula.
#
# 3. score combination: cls2 output is raw logit. Must apply sigmoid before
#    combining with cls1 softmax score (which is already in [0,1]).
#    Old code used cls2.view(-1) directly — this gives logits mixed with
#    probabilities, biasing the combined score unpredictably.
#
# 4. _convert_score: cls1 output is (B, 2, H, W) from the model.
#    Permute → (H*W, 2) → softmax → take channel 1 (fg prob). Correct.
#
# 5. scale_z in init(): was set to s_z (raw pixel size of template crop),
#    but it was used later as a fallback "original scale". This is fine as-is.
#
# 6. Hanning window size must match score_size (11 for 287-input model).
#    Old code computed it from cfg.TRAIN.OUTPUT_SIZE which is already 11 —
#    consistent, no change needed but clarified.
# ============================================================

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

        # FIX #1: score_size must match training output size (11 for 287 input)
        self.score_size = cfg.TRAIN.OUTPUT_SIZE   # 11

        hanning = np.hanning(self.score_size)
        self.window = np.outer(hanning, hanning).flatten()

    def _inverse_transform(self, x):
        """Inverse tanh: maps (-1,1) → R. Clips to avoid log(0)."""
        x = np.clip(x, -0.9999, 0.9999)
        return (np.log(1 + x) - np.log(1 - x)) / 2.0

    def generate_anchor(self, loc_map):
        """
        Decode (1, 4, H, W) localization output to (H*W, 4) center+size anchors.

        Returns: (4, H*W) array — rows are [cx, cy, w, h] in search-image coords
                 (search-image is SEARCH_SIZE × SEARCH_SIZE)
        """
        size   = self.score_size         # 11
        stride = cfg.ANCHOR.STRIDE       # 16
        offset = 63                      # centering offset for 287-px search image
        half_search = cfg.TRAIN.SEARCH_SIZE // 2   # 143

        # 1-D coordinate arrays (centered)
        xs = stride * np.arange(size) + offset - half_search   # (11,)
        ys = stride * np.arange(size) + offset - half_search   # (11,)

        # Full grid (column = x, row = y)
        x_grid = np.tile(xs, size)            # (121,) x coords
        y_grid = np.repeat(ys, size)          # (121,) y coords

        # Decode shape deltas from tanh-space
        loc_np = loc_map[0].cpu().detach().numpy()   # (4, 11, 11)
        shape_deltas = self._inverse_transform(loc_np) * half_search  # real distances

        # Flatten spatial dims: (4, 121)
        left   = shape_deltas[0].flatten()
        right  = shape_deltas[1].flatten()
        top    = shape_deltas[2].flatten()
        bottom = shape_deltas[3].flatten()

        # Width and height from FCOS-style left/right/top/bottom
        w  = left + right
        h  = top + bottom

        # Center coordinates (back to absolute search-image frame)
        cx = x_grid - left + w / 2.0   # = x_grid + (right - left) / 2
        cy = y_grid - top  + h / 2.0

        anchor = np.zeros((4, size * size))
        anchor[0, :] = cx   # cx relative to search center
        anchor[1, :] = cy   # cy relative to search center
        anchor[2, :] = w
        anchor[3, :] = h
        return anchor

    def _convert_score(self, cls1):
        """
        Convert cls1 (B, 2, H, W) logits → (H*W,) foreground probability.
        """
        # Reshape to (H*W, 2), apply softmax, take fg channel
        score = cls1.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _convert_cls2_score(self, cls2):
        """
        Convert cls2 (B, 1, H, W) raw logits → (H*W,) probability via sigmoid.
        FIX #3: Old code used raw logits — now correctly applies sigmoid.
        """
        score = torch.sigmoid(cls2).view(-1).cpu().detach().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        """Clamp predicted box to image boundary."""
        cx     = max(0, min(cx, boundary[1]))
        cy     = max(0, min(cy, boundary[0]))
        width  = max(10, min(width,  boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        Initialize tracker on first frame.

        Args:
            img:  BGR image (H, W, 3)
            bbox: [x, y, width, height] (0-based, top-left)
        """
        self.image = img
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # Context-padded template crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.scale_z = s_z   # save for fallback

        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        Track in a new frame.

        Returns:
            dict with 'bbox' [x, y, w, h] and 'best_score'
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)

        # Fallback for extreme scale drift
        if self.size[0] * self.size[1] > 0.5 * img.shape[0] * img.shape[1]:
            s_z = self.scale_z

        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        # FIX #1: Use SEARCH_SIZE (287) at inference to match training input size
        # This ensures the feature map is 11×11 as expected
        search_size = cfg.TRAIN.SEARCH_SIZE   # 287  (was cfg.TRACK.INSTANCE_SIZE=255)
        s_x = s_z * (search_size / cfg.TRACK.EXEMPLAR_SIZE)

        x_crop = self.get_subwindow(img, self.center_pos,
                                    search_size,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        # Decode predictions
        pred_bbox = self.generate_anchor(outputs['loc'])   # (4, 121) [cx,cy,w,h]

        # FIX #3: Apply sigmoid to cls2 before combining scores
        score1 = self._convert_score(outputs['cls1']) * cfg.TRACK.w2
        score2 = self._convert_cls2_score(outputs['cls2']) * cfg.TRACK.w3
        score = (score1 + score2) / 2.0

        # Scale and aspect-ratio penalty
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
        pscore = penalty * score
        pscore = (pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE)
                  + self.window * cfg.TRACK.WINDOW_INFLUENCE)

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z   # [cx, cy, w, h] in image coords
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        # Smooth update of position and size
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width  = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height,
                                                img.shape[:2])
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        return {
            'bbox': [cx - width / 2, cy - height / 2, width, height],
            'best_score': float(score[best_idx])
        }