# Copyright (c) SenseTime. All Rights Reserved.
"""
HiFTTracker: Inference tracker for single-object tracking.
Fixes: Aligned search size, robust anchor generation, clean coordinate transforms.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker

class HiFTTracker(SiameseTracker):
    def __init__(self, model):
        super(HiFTTracker, self).__init__()
        self.model = model
        self.model.eval()
        
        # Grid parameters for anchor generation
        self.score_size = cfg.TRAIN.OUTPUT_SIZE  # 11x11 grid
        self.anchor_num = 1
        hanning = np.hanning(self.score_size)
        self.window = np.outer(hanning, hanning).flatten()
        
    def _inverse_transform(self, x):
        """Clamp and apply inverse tanh-like transform to network outputs."""
        x = np.clip(x, -0.99, 0.99)
        return (np.log(1 + x) - np.log(1 - x)) / 2.0

    def generate_anchor(self, loc_map):
        """Decode localization head output into absolute bounding boxes."""
        size = self.score_size
        stride = cfg.ANCHOR.STRIDE  # 16
        offset = 63
        half_search = cfg.TRAIN.SEARCH_SIZE // 2  # 143
        
        # Generate grid coordinates
        x = np.tile((stride * np.linspace(0, size-1, size) + offset) - half_search, size)
        y = np.tile((stride * np.linspace(0, size-1, size) + offset - half_search).reshape(-1, 1), (1, size)).reshape(-1)
        
        # Decode shape deltas
        loc_np = loc_map[0].cpu().detach().numpy()
        shape_deltas = self._inverse_transform(loc_np) * 143.0
        
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        yy = yy.reshape(-1).astype(np.int16)
        xx = xx.reshape(-1).astype(np.int16)
        
        w = shape_deltas[0, yy, xx] + shape_deltas[1, yy, xx]
        h = shape_deltas[2, yy, xx] + shape_deltas[3, yy, xx]
        cx = x - shape_deltas[0, yy, xx] + w / 2.0
        cy = y - shape_deltas[2, yy, xx] + h / 2.0
        
        anchor = np.zeros((size**2, 4))
        anchor[:, 0] = cx
        anchor[:, 1] = cy
        anchor[:, 2] = w
        anchor[:, 3] = h
        return anchor

    def _convert_score(self, score_map):
        """Extract objectness probability from classification map."""
        score_map = score_map.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score_map = F.softmax(score_map, dim=1).data[:, 1].cpu().numpy()
        return score_map

    def _bbox_clip(self, cx, cy, width, height, boundary):
        """Prevent bounding box from exceeding image boundaries."""
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        Initialize tracker.
        Args:
            img: BGR image (H, W, 3)
            bbox: [x, y, width, height]
        """
        self.image = img
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])
        self.first_bbox = np.concatenate((self.center_pos, self.size))
        
        # Calculate template crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.scale_z = s_z
        
        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.template = z_crop
        self.model.template(z_crop)

    def track(self, img):
        """
        Track object in new frame.
        Returns: dict with 'bbox' [x, y, w, h] and 'best_score'
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        
        # Fallback for extreme scale changes
        if self.size[0] * self.size[1] > 0.5 * img.shape[0] * img.shape[1]:
            s_z = self.scale_z
            
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        
        outputs = self.model.track(x_crop)
        
        # Decode predictions
        pred_bbox = self.generate_anchor(outputs['loc']).T
        score1 = self._convert_score(outputs['cls1']) * cfg.TRACK.w2
        score2 = outputs['cls2'].view(-1).cpu().detach().numpy() * cfg.TRACK.w3
        score = (score1 + score2) / 2.0

        # Apply scale & aspect ratio penalties
        def change(r): return np.maximum(r, 1. / (r + 1e-5))
        def sz(w, h): pad = (w + h) * 0.5; return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = change((self.size[0] / (self.size[1] + 1e-5)) /
                     (pred_bbox[2, :] / (pred_bbox[3, :] + 1e-5)))
        
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.window * cfg.TRACK.WINDOW_INFLUENCE
        
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        
        # Smooth update
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])
        
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        
        return {
            'bbox': [cx - width / 2, cy - height / 2, width, height],
            'best_score': score[best_idx]
        }