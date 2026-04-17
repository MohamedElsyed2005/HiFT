# ============================================================
# BUG FIXES IN THIS FILE:
# 1. OUTPUT_SIZE was inconsistent — AnchorTarget.get() was called with
#    cfg.TRAIN.OUTPUT_SIZE (which was 21 in base config, 11 in YAML).
#    The label grids are now consistently sized to OUTPUT_SIZE=11.
# 2. OUTPUTFEATURE_SIZE is used for reshape in label generation — must equal
#    OUTPUT_SIZE. These were decoupled and could cause shape mismatch.
#    Fix: use OUTPUT_SIZE everywhere inside get() for grid generation.
# 3. The pr grid generation had a bug: pre was shaped (size,1) but was being
#    multiplied into a (size^2,2) array incorrectly. Fixed indexing.
# 4. weightxff and labelxff used cfg.TRAIN.OUTPUTFEATURE_SIZE for reshape —
#    changed to cfg.TRAIN.OUTPUT_SIZE to be consistent.
# ============================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from pysot.core.config import cfg
from pysot.utils.bbox import IoU


class AnchorTarget():
    def __init__(self):
        return

    def select(self, position, keep_num=16):
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num

    def get(self, bbox, size):
        """
        Generate training labels for a single (template, search) pair.

        Args:
            bbox: Corner namedtuple (x1, y1, x2, y2) in search-image coordinates
            size: output feature map size (= cfg.TRAIN.OUTPUT_SIZE = 11)

        Returns:
            labelcls1  : (1, size, size)  — hard pos/neg label for cls1 head
            labelxff   : (4, size, size)  — regression target (tanh-encoded)
            labelcls2  : (1, size, size)  — soft Gaussian label for BCE cls2 head
            weightxff  : (1, size, size)  — per-cell regression weight
        """
        # FIX: Use OUTPUT_SIZE (== size arg) consistently throughout.
        # Old code used cfg.TRAIN.OUTPUTFEATURE_SIZE for reshape which could
        # differ from the 'size' argument causing silent shape corruption.

        labelcls1 = np.zeros((1, size, size)) - 1

        # Build physical coordinate grid for the search region
        # Each grid cell maps to stride=16 pixels in the 287x287 search image
        stride = cfg.ANCHOR.STRIDE  # 16
        offset = 63  # (127-1)/2 = 63, centering offset

        # pr[:,0] = x-coordinates, pr[:,1] = y-coordinates (in search-image space)
        # Centered so that 0 = center of search image
        xs = stride * np.arange(size) + offset  # absolute pixel coords
        ys = stride * np.arange(size) + offset

        pr = np.zeros((size * size, 2))
        # FIX: correct meshgrid — outer = y (row), inner = x (col)
        pr[:, 0] = np.tile(xs, size)                    # x coords, shape (size^2,)
        pr[:, 1] = np.repeat(ys, size)                  # y coords, shape (size^2,)

        labelxff = np.zeros((4, size, size), dtype=np.float32)
        labelcls2 = np.zeros((1, size, size))
        weightxff = np.zeros((1, size, size))

        target = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2])

        # Map target corners to feature-map indices
        index2 = np.int32(np.minimum(size - 1, np.maximum(0, (target - offset) / stride)))
        w = int(index2[2] - index2[0] + 1)
        h = int(index2[3] - index2[1] + 1)

        label_range = cfg.TRAIN.LABEL_RANGE  # 4

        # --- Build regression weight map (Gaussian-like, centered on target) ---
        for ii in np.arange(size):
            for jj in np.arange(size):
                weightxff[0, ii, jj] = (
                    ((ii - (index2[1] + index2[3]) / 2) * label_range) ** 2
                    + ((jj - (index2[0] + index2[2]) / 2) * label_range) ** 2
                )

        radius_sq = ((w // 2 + h // 2) * label_range / 1.5) ** 2
        in_mask = weightxff < radius_sq
        se = weightxff[in_mask]

        # Normalize weights inside the radius to [0,1] then invert
        if se.max() > se.min():
            weightxff[in_mask] = 1 - (se - se.min()) / (se.max() - se.min() + 1e-4)
        else:
            weightxff[in_mask] = 1.0
        weightxff[~in_mask] = 0.0

        # Boost top-scoring positive cells
        pos = np.where(weightxff.squeeze() > 0.8)
        num = len(pos[0])
        if num > 0:
            pos, _ = self.select(pos, max(1, num // 4))
            weightxff[:, pos[0], pos[1]] = 1.5

        # Re-compute index (same formula, kept separate for clarity)
        index = np.int32(np.minimum(size - 1, np.maximum(0, (target - offset) / stride)))
        w = int(index[2] - index[0] + 1)
        h = int(index[3] - index[1] + 1)

        # --- Build soft classification label (Gaussian) for BCE cls2 head ---
        for ii in np.arange(size):
            for jj in np.arange(size):
                labelcls2[0, ii, jj] = (
                    ((ii - (index[1] + index[3]) / 2) * label_range) ** 2
                    + ((jj - (index[0] + index[2]) / 2) * label_range) ** 2
                )

        radius_sq2 = ((w // 2 + h // 2) * label_range / 1.2) ** 2
        in_mask2 = labelcls2 < radius_sq2
        see = labelcls2[in_mask2]

        weightcls3 = np.zeros((1, size, size))
        weightcls3[in_mask2] = 1.0

        if see.max() > see.min():
            labelcls2[in_mask2] = 1 - (see - see.min()) / (see.max() - see.min() + 1e-4)
        else:
            labelcls2[in_mask2] = 1.0
        labelcls2 = labelcls2 * weightcls3  # zero out outside radius

        # --- Build regression label (tanh-encoded FCOS-style offsets) ---
        def con(x):
            """Tanh: maps R -> (-1, 1). Encodes large offsets compactly."""
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        half_search = cfg.TRAIN.SEARCH_SIZE // 2  # 143

        # FIX: reshape using 'size' not OUTPUTFEATURE_SIZE to avoid shape bug
        px = pr[:, 0].reshape(size, size)  # (size, size) x physical coords
        py = pr[:, 1].reshape(size, size)  # (size, size) y physical coords

        # Centered coordinates (search center = 0)
        px_c = px - half_search
        py_c = py - half_search

        # FCOS-style: (left, right, top, bottom) distances from grid cell to bbox
        labelxff[0, :, :] = px_c - (target[0] - half_search)   # left
        labelxff[1, :, :] = (target[2] - half_search) - px_c   # right
        labelxff[2, :, :] = py_c - (target[1] - half_search)   # top
        labelxff[3, :, :] = (target[3] - half_search) - py_c   # bottom

        labelxff = con(labelxff / half_search)

        # --- Build hard classification label (cls1 cross-entropy) ---
        # Inner box = foreground (1), outer ring = ignored (-2), rest = neg (then →0)
        labelcls1[0,
                  index[1] - h // 4: index[3] + 1 + h // 4,
                  index[0] - w // 4: index[2] + 1 + w // 4] = -2
        labelcls1[0,
                  index[1] + h // 4: index[3] + 1 - h // 4,
                  index[0] + w // 4: index[2] + 1 - w // 4] = 1

        # Sample negatives to balance with positives (ratio ~2.5:1)
        neg2 = np.where(labelcls1.squeeze() == -1)
        n_pos = len(np.where(labelcls1 == 1)[0])
        n_neg_keep = max(1, int(n_pos * 2.5))
        neg2, _ = self.select(neg2, n_neg_keep)
        labelcls1[:, neg2[0], neg2[1]] = 0   # turn sampled negatives to 0 (active)

        return labelcls1, labelxff, labelcls2, weightxff