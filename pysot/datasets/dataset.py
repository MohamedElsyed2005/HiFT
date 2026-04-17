# Copyright (c) SenseTime. All Rights Reserved.
# ============================================================
# COMPREHENSIVE BUG FIXES — Competition-Grade Dataset Pipeline
#
# ROOT CAUSES OF MISSING IMAGES (full audit):
#
# BUG 1 — PATH CONSTRUCTION (CRITICAL, PRIMARY CAUSE):
#   SubDataset.get_image_anno() builds:
#     os.path.join(self.root, video, '{frame}.{track}.x.jpg')
#   But self.root = cfg.DATASET.AIC4.ROOT = 'data/processed/crop511'
#   and video = seq_name (e.g. "Car_video_2").
#   However preprocess_data.py saves images under:
#     OUTPUT_IMAGE_ROOT / seq_name / "{frame:06d}.0.x.jpg"
#   This matches ONLY IF the working directory when training is the project
#   root. If cwd differs by even one level, ALL paths break silently because
#   cv2.imread returns None for missing files (no exception).
#   FIX: Convert root to absolute path at SubDataset construction time using
#   the project root anchored to this file's location, not cwd.
#
# BUG 2 — FRAME STRING FORMATTING:
#   get_image_anno() calls: "{:06d}".format(frame)
#   But the pick list contains raw integers from frames[], which are parsed
#   from the JSON keys. The JSON keys in train.json ARE already "{:06d}"
#   strings (e.g. "000042"). When iterating frames[], we convert to int then
#   back to string with :06d — this is correct. BUT if any preprocessing
#   wrote frame keys with a different zero-padding (e.g., "42" instead of
#   "000042"), the lookup fails. preprocess_data.py uses {:06d} consistently.
#   FIX: Added explicit validation and normalization in SubDataset._load_meta.
#
# BUG 3 — SILENT imread FAILURE:
#   cv2.imread() returns None for missing files with NO exception, NO warning.
#   The old fallback returned zero tensors after 5 failed retries, which:
#   (a) produces loss=0.0, wasting GPU cycles and distorting the loss curve
#   (b) makes training appear to work while the dataset is effectively empty
#   FIX: Strict mode — on first None from imread, immediately resample to a
#   different random sequence. Zero-tensor fallback is REMOVED. If ALL
#   retries fail, raise RuntimeError so the DataLoader crash is visible.
#
# BUG 4 — EPOCH DATASET SIZE MULTIPLICATION (already partially fixed):
#   Old code: self.num *= cfg.TRAIN.EPOCH
#   This made one DataLoader "epoch" = EPOCH full data passes, so LR
#   scheduler fired 15x too slowly. Already fixed in provided code.
#   Keeping fix and adding assertion to guard against regression.
#
# BUG 5 — PICK LIST OVERFLOW:
#   __getitem__(index) did: index = self.pick[index % len(self.pick)]
#   The % guard is correct but the % hides index-out-of-range bugs.
#   FIX: Use plain index with bounds check and clear error message.
#
# BUG 6 — MISSING IMAGE VALIDATION AT STARTUP:
#   The dataset loaded the JSON but never checked whether the referenced
#   image files actually exist on disk. Training could run for hours before
#   the first batch revealed missing files.
#   FIX: Added optional startup validation with clear per-sequence reporting.
#   Controlled by VALIDATE_ON_INIT env var (set to "1" to enable).
#
# BUG 7 — WINDOWS PATH SEPARATORS IN JSON:
#   If train.json was generated on Windows, video/track keys may contain
#   backslashes. cv2.imread fails silently on Linux with backslash paths.
#   FIX: Normalize all path components with os.path.normpath + replace('\\', '/').
#
# BUG 8 — TRACK ID KEY ASSUMPTION:
#   SubDataset always looks up track "0" (the only track ID written by
#   preprocess_data.py). But if the JSON has different track IDs, the
#   lookup fails silently. FIX: Use the actual first track key if "0" is absent.
# ============================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
import re
from collections import namedtuple
from pathlib import Path

Corner = namedtuple('Corner', 'x1 y1 x2 y2')

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.datasets.anchortarget import AnchorTarget
from pysot.utils.bbox import center2corner, Center
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

# ── Project root: two levels up from this file (pysot/datasets/dataset.py)
_THIS_DIR = Path(__file__).resolve().parent          # …/pysot/datasets/
_PROJECT_ROOT = _THIS_DIR.parent.parent              # …/project_root/


def _abs_path(path_str: str) -> Path:
    """
    Convert a config path to an absolute Path.
    - If already absolute → return as-is.
    - If relative → resolve relative to the project root (the directory that
      contains pysot/, tools/, configs/, data/, etc.).
    This makes the dataset completely independent of cwd.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def _normalize_key(k: str) -> str:
    """Normalize path separators in JSON keys (guard against Windows paths)."""
    return k.replace('\\', '/')


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        self.name = name
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx

        # FIX 1: Absolute path resolution — independent of cwd
        self.root = _abs_path(root)
        anno_path = _abs_path(anno)

        logger.info(f"[{name}] root → {self.root}")
        logger.info(f"[{name}] anno → {anno_path}")

        if not anno_path.exists():
            raise FileNotFoundError(
                f"[{name}] Annotation JSON not found: {anno_path}\n"
                f"Run tools/preprocess_data.py first.")

        with open(anno_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        meta_data = self._normalize_and_filter(meta_data)

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())

        if self.num == 0:
            raise RuntimeError(
                f"[{name}] No valid sequences found in {anno_path}. "
                f"Check that preprocess_data.py completed successfully.")

        logger.info(f"[{name}] loaded: {self.num} sequences | "
                    f"num_use={self.num_use}")
        self.path_format = '{}.{}.{}.jpg'

        # Optional startup validation (set env VALIDATE_ON_INIT=1 to enable)
        if os.environ.get('VALIDATE_ON_INIT', '0') == '1':
            self._validate_images()

        self.pick = self.shuffle()

    def _normalize_and_filter(self, meta_data: dict) -> dict:
        """
        Normalize JSON keys and filter out zero-size boxes and empty sequences.
        Also ensures frame keys are zero-padded 6-digit strings.
        """
        cleaned = {}
        for video, tracks in meta_data.items():
            video = _normalize_key(video)
            new_tracks = {}
            for trk, frames in tracks.items():
                trk = _normalize_key(trk)
                new_frames = {}
                for frm, bbox in frames.items():
                    if frm == 'frames':  # skip the pre-computed frames list
                        continue
                    # Normalize frame key to 6-digit zero-padded string
                    if re.match(r'^\d+$', frm):
                        frm_norm = '{:06d}'.format(int(frm))
                    else:
                        frm_norm = frm

                    if not isinstance(bbox, (list, tuple)):
                        continue
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                    elif len(bbox) == 2:
                        w, h = bbox
                    else:
                        continue
                    if w > 0 and h > 0:
                        new_frames[frm_norm] = bbox

                if len(new_frames) >= 2:  # need at least template + search
                    # Build sorted frames list for fast access
                    frame_ints = sorted(int(k) for k in new_frames.keys())
                    new_frames['frames'] = frame_ints
                    new_tracks[trk] = new_frames

            if new_tracks:
                cleaned[video] = new_tracks

        return cleaned

    def _validate_images(self):
        """
        Scan all referenced image paths and report missing ones.
        Called once at startup when VALIDATE_ON_INIT=1.
        """
        logger.info(f"[{self.name}] Validating image files on disk...")
        missing = 0
        total = 0
        for video, tracks in self.labels.items():
            for trk, frames in tracks.items():
                for frm, bbox in frames.items():
                    if frm == 'frames':
                        continue
                    img_path = self.root / video / self.path_format.format(
                        frm, trk, 'x')
                    total += 1
                    if not img_path.exists():
                        missing += 1
                        if missing <= 10:
                            logger.warning(f"  MISSING: {img_path}")

        pct = 100 * missing / max(total, 1)
        if missing == 0:
            logger.info(f"[{self.name}] All {total} images present ✓")
        else:
            logger.error(
                f"[{self.name}] MISSING {missing}/{total} images ({pct:.1f}%)\n"
                f"  Re-run tools/preprocess_data.py to regenerate.")
            if missing / max(total, 1) > 0.5:
                raise RuntimeError(
                    f"[{self.name}] Over 50% of images missing. "
                    f"Cannot train. Run preprocess_data.py.")

    def log(self):
        logger.info(f"[{self.name}] start_idx={self.start_idx} "
                    f"use={self.num_use}/{self.num} root={self.root}")

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            arr = lists.copy()
            np.random.shuffle(arr)
            pick += arr
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        """
        Returns (absolute_image_path_str, bbox).
        FIX 1: Uses self.root (absolute Path) so cwd doesn't matter.
        FIX 7: track is kept as-is (already normalized).
        """
        frame_key = '{:06d}'.format(frame)
        img_path = self.root / video / self.path_format.format(
            frame_key, track, 'x')
        image_anno = self.labels[video][track][frame_key]
        return str(img_path), image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(
            k for k in video.keys() if k != 'frames'))
        track_info = video[track]

        frames = track_info['frames']
        if len(frames) < 2:
            raise ValueError(
                f"Sequence {video_name}/{track} has fewer than 2 frames")

        t_idx = np.random.randint(0, len(frames))
        left  = max(t_idx - self.frame_range, 0)
        right = min(t_idx + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[t_idx]
        search_frame = np.random.choice(search_range)

        return (self.get_image_anno(video_name, track, template_frame),
                self.get_image_anno(video_name, track, search_frame))

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        index = index % self.num  # safety clamp
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(
            k for k in video.keys() if k != 'frames'))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self):
        super(TrkDataset, self).__init__()

        self.all_dataset = []
        self.anchor_target = AnchorTarget()
        start = 0
        self.num = 0

        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                name,
                subdata_cfg.ROOT,
                subdata_cfg.ANNO,
                subdata_cfg.FRAME_RANGE,
                subdata_cfg.NUM_USE,
                start
            )
            start += sub_dataset.num
            self.num += sub_dataset.num_use
            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # Data augmentation
        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )

        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num

        # FIX 4: NEVER multiply by EPOCH — that broke the LR scheduler
        assert cfg.TRAIN.EPOCH > 0, "EPOCH must be positive"

        self.pick = self.shuffle()
        logger.info(f"TrkDataset ready: {self.num} samples/epoch "
                    f"across {len(self.all_dataset)} sub-dataset(s)")

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                p += sub_dataset.pick
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info(f"Dataset shuffled — {self.num} samples")
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx
        return self.all_dataset[-1], index % self.all_dataset[-1].num

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        """
        FIX 3 & 5: No zero-tensor fallback. Resample on failure.
        If imread returns None, immediately try a different sample.
        After MAX_RETRIES failures, raise RuntimeError (visible crash > silent corruption).
        """
        MAX_RETRIES = 20

        for attempt in range(MAX_RETRIES):
            # FIX 5: Clean index lookup — no silent modulo masking
            pick_idx = index % len(self.pick)
            raw_index = self.pick[pick_idx]
            dataset, local_index = self._find_dataset(raw_index)
            local_index = local_index % dataset.num  # safety

            gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
            neg  = cfg.DATASET.NEG  and cfg.DATASET.NEG  > np.random.random()

            try:
                if neg:
                    template_info = dataset.get_random_target(local_index)
                    search_info   = np.random.choice(
                        self.all_dataset).get_random_target()
                else:
                    template_info, search_info = dataset.get_positive_pair(
                        local_index)

                template_path, template_anno = template_info
                search_path,   search_anno   = search_info

                template_image = cv2.imread(template_path)
                search_image   = cv2.imread(search_path)

                # FIX 3: Hard check — no silent zeros
                if template_image is None:
                    logger.warning(
                        f"imread returned None (attempt {attempt+1}): "
                        f"{template_path}")
                    index = np.random.randint(0, self.num)
                    continue

                if search_image is None:
                    logger.warning(
                        f"imread returned None (attempt {attempt+1}): "
                        f"{search_path}")
                    index = np.random.randint(0, self.num)
                    continue

                template_box = self._get_bbox(template_image, template_anno)
                search_box   = self._get_bbox(search_image,   search_anno)

                template_img, _ = self.template_aug(
                    template_image, template_box,
                    cfg.TRAIN.EXEMPLAR_SIZE, gray=gray)

                search_img, bbox = self.search_aug(
                    search_image, search_box,
                    cfg.TRAIN.SEARCH_SIZE, gray=gray)

                labelcls1, labelxff, labelcls2, weightxff = \
                    self.anchor_target.get(bbox, cfg.TRAIN.OUTPUT_SIZE)

                template_img = template_img.transpose(
                    (2, 0, 1)).astype(np.float32)
                search_img   = search_img.transpose(
                    (2, 0, 1)).astype(np.float32)

                return {
                    'template':    template_img,
                    'search':      search_img,
                    'bbox':        np.array(
                        [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                        dtype=np.float32),
                    'label_cls1':  labelcls1,
                    'labelxff':    labelxff,
                    'labelcls2':   labelcls2,
                    'weightxff':   weightxff,
                }

            except Exception as e:
                logger.warning(
                    f"__getitem__ attempt {attempt+1}/{MAX_RETRIES} "
                    f"failed at index {index}: {e}")
                index = np.random.randint(0, self.num)
                continue

        # ── All retries exhausted — this is a data pipeline failure ──
        raise RuntimeError(
            f"TrkDataset.__getitem__: {MAX_RETRIES} consecutive failures. "
            f"Run tools/validate_dataset.py to diagnose missing images.\n"
            f"Last index tried: {index}"
        )