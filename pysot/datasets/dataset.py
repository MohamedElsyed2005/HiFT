# Copyright (c) SenseTime. All Rights Reserved.
# ============================================================
# ADDITIONAL FIX IN THIS VERSION (on top of existing fixes):
#
# ROOT CAUSE OF SILENT TRAINING WITH WRONG JSON:
# preprocess_data.py was writing the RAW MANIFEST METADATA (video_path,
# annotation_path, n_frames etc.) to manifest_train.json, NOT the crop
# annotations.  If cfg.DATASET.AIC4.ANNO was pointing to a stale or wrong
# file, dataset.py would silently succeed (because the JSON parsed fine)
# but _normalize_and_filter would produce 0 sequences (because the keys
# inside have "video_path", "annotation_path" etc., not frame dicts).
# The dataset would then have self.num==0 which raised RuntimeError — but
# only if the code reached the assertion.  In some code paths the error
# was swallowed.
#
# FIX: Added _validate_json_structure() called at SubDataset init.
# It checks that at least one sequence has at least one frame entry that
# looks like a bbox ([x1,y1,x2,y2]) and raises a clear RuntimeError if
# the JSON appears to be a raw manifest instead of crop annotations.
#
# All other existing fixes retained:
# - Absolute path resolution (FIX 1)
# - Frame string normalization (FIX 2)
# - No silent imread fallback (FIX 3)
# - No epoch multiplication (FIX 4)
# - Clean index lookup (FIX 5)
# - Startup image validation (FIX 6)
# - Windows path separators (FIX 7)
# - Track ID key assumption (FIX 8)
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

_THIS_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent


def _abs_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def _normalize_key(k: str) -> str:
    return k.replace('\\', '/')


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        self.name       = name
        self.frame_range = frame_range
        self.num_use    = num_use
        self.start_idx  = start_idx
        self.root       = _abs_path(root)
        anno_path       = _abs_path(anno)

        logger.info(f"[{name}] root → {self.root}")
        logger.info(f"[{name}] anno → {anno_path}")

        if not anno_path.exists():
            raise FileNotFoundError(
                f"[{name}] Annotation JSON not found: {anno_path}\n"
                f"Run tools/preprocess_data.py first.")

        with open(anno_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        # ── NEW: validate that this JSON is a crop-annotation file ──────────
        self._validate_json_structure(meta_data, anno_path)

        meta_data = self._normalize_and_filter(meta_data)

        self.labels   = meta_data
        self.num      = len(self.labels)
        self.num_use  = self.num if self.num_use == -1 else self.num_use
        self.videos   = list(meta_data.keys())

        if self.num == 0:
            raise RuntimeError(
                f"[{name}] No valid sequences found in {anno_path}.\n"
                f"Check that preprocess_data.py completed successfully and\n"
                f"that cfg.DATASET.{name}.ANNO points to the CROP ANNOTATION\n"
                f"JSON (data/processed/train.json), NOT to manifest_train.json.")

        logger.info(f"[{name}] loaded: {self.num} sequences | num_use={self.num_use}")
        self.path_format = '{}.{}.{}.jpg'

        if os.environ.get('VALIDATE_ON_INIT', '0') == '1':
            self._validate_images()

        self.pick = self.shuffle()

    # ── NEW: structural validation ─────────────────────────────────────────
    def _validate_json_structure(self, meta_data: dict, anno_path: Path):
        """
        Raise a clear error if this JSON looks like a raw manifest rather than
        a crop-annotation file.

        Crop-annotation format (expected):
            {"seq_key": {"0": {"000001": [x1,y1,x2,y2], ...}}}

        Raw manifest format (wrong):
            {"seq_key": {"dataset": "...", "video_path": "...", "n_frames": ...}}
        """
        if not meta_data:
            return  # empty — caught by num==0 check below

        # Sample first sequence
        first_seq = next(iter(meta_data.values()))
        if not isinstance(first_seq, dict):
            raise RuntimeError(
                f"JSON format error in {anno_path}:\n"
                f"  Expected dict of track dicts, got {type(first_seq)}.\n"
                f"  This looks like a raw manifest file, not a crop annotation.\n"
                f"  Run tools/preprocess_data.py to regenerate train.json.")

        # Check if it has raw manifest keys instead of track-id keys
        raw_manifest_keys = {'video_path', 'annotation_path', 'n_frames',
                              'dataset', 'seq_name', 'native_fps'}
        seq_keys = set(first_seq.keys())
        overlap  = seq_keys & raw_manifest_keys
        if overlap:
            raise RuntimeError(
                f"JSON format error in {anno_path}:\n"
                f"  Found raw manifest keys: {overlap}\n"
                f"  This file contains original manifest metadata, NOT crop annotations.\n"
                f"  The correct file is data/processed/train.json (crop annotation JSON).\n"
                f"  Run tools/preprocess_data.py (fixed version) to regenerate it.\n"
                f"  Make sure cfg.DATASET.AIC4.ANNO = 'data/processed/train.json'.")

        # Spot-check: first track's first frame should be a bbox list/tuple
        first_track = next(iter(first_seq.values()))
        if not isinstance(first_track, dict):
            raise RuntimeError(
                f"JSON format error in {anno_path}:\n"
                f"  Expected track dict, got {type(first_track)}.\n"
                f"  Run tools/preprocess_data.py to regenerate train.json.")

        for frm_key, val in first_track.items():
            if frm_key == 'frames':
                continue
            if not isinstance(val, (list, tuple)) or len(val) != 4:
                raise RuntimeError(
                    f"JSON format error in {anno_path}:\n"
                    f"  Frame bbox should be [x1,y1,x2,y2], got: {val}\n"
                    f"  Run tools/preprocess_data.py to regenerate train.json.")
            break  # only check first frame

        logger.info(f"[{self.name}] JSON structure validated ✓")

    def _normalize_and_filter(self, meta_data: dict) -> dict:
        cleaned = {}
        for video, tracks in meta_data.items():
            video = _normalize_key(video)
            new_tracks = {}
            for trk, frames in tracks.items():
                trk = _normalize_key(trk)
                new_frames = {}
                for frm, bbox in frames.items():
                    if frm == 'frames':
                        continue
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
                if len(new_frames) >= 2:
                    frame_ints = sorted(int(k) for k in new_frames.keys())
                    new_frames['frames'] = frame_ints
                    new_tracks[trk] = new_frames
            if new_tracks:
                cleaned[video] = new_tracks
        return cleaned

    def _validate_images(self):
        logger.info(f"[{self.name}] Validating image files on disk...")
        missing = 0
        total   = 0
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
                f"[{self.name}] MISSING {missing}/{total} ({pct:.1f}%)\n"
                f"  Re-run tools/preprocess_data.py.")
            if missing / max(total, 1) > 0.5:
                raise RuntimeError(
                    f"[{self.name}] Over 50% images missing. "
                    f"Run preprocess_data.py.")

    def log(self):
        logger.info(f"[{self.name}] start={self.start_idx} "
                    f"use={self.num_use}/{self.num} root={self.root}")

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick  = []
        while len(pick) < self.num_use:
            arr = lists.copy()
            np.random.shuffle(arr)
            pick += arr
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame_key  = '{:06d}'.format(frame)
        img_path   = self.root / video / self.path_format.format(
            frame_key, track, 'x')
        image_anno = self.labels[video][track][frame_key]
        return str(img_path), image_anno

    def get_positive_pair(self, index):
        video_name  = self.videos[index]
        video       = self.labels[video_name]
        track       = np.random.choice(
            list(k for k in video.keys() if k != 'frames'))
        track_info  = video[track]
        frames      = track_info['frames']
        if len(frames) < 2:
            raise ValueError(
                f"Sequence {video_name}/{track} has fewer than 2 frames")
        t_idx        = np.random.randint(0, len(frames))
        left         = max(t_idx - self.frame_range, 0)
        right        = min(t_idx + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[t_idx]
        search_frame   = np.random.choice(search_range)
        return (self.get_image_anno(video_name, track, template_frame),
                self.get_image_anno(video_name, track, search_frame))

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        index      = index % self.num
        video_name = self.videos[index]
        video      = self.labels[video_name]
        track      = np.random.choice(
            list(k for k in video.keys() if k != 'frames'))
        track_info = video[track]
        frames     = track_info['frames']
        frame      = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self):
        super(TrkDataset, self).__init__()
        self.all_dataset   = []
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
            start    += sub_dataset.num
            self.num += sub_dataset.num_use
            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT, cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,  cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR)
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT, cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,  cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR)

        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num

        assert cfg.TRAIN.EPOCH > 0, "EPOCH must be positive"

        self.pick = self.shuffle()
        logger.info(f"TrkDataset ready: {self.num} samples/epoch across "
                    f"{len(self.all_dataset)} sub-dataset(s)")

    def shuffle(self):
        pick = []
        m    = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                p += sub_dataset.pick
            np.random.shuffle(p)
            pick += p
            m     = len(pick)
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
        exemplar_size  = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z  = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox   = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        MAX_RETRIES = 20

        for attempt in range(MAX_RETRIES):
            pick_idx  = index % len(self.pick)
            raw_index = self.pick[pick_idx]
            dataset, local_index = self._find_dataset(raw_index)
            local_index = local_index % dataset.num

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

                if template_image is None:
                    logger.warning(
                        f"imread None (attempt {attempt+1}): {template_path}")
                    index = np.random.randint(0, self.num)
                    continue

                if search_image is None:
                    logger.warning(
                        f"imread None (attempt {attempt+1}): {search_path}")
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

                template_img = template_img.transpose((2, 0, 1)).astype(np.float32)
                search_img   = search_img.transpose((2, 0, 1)).astype(np.float32)

                return {
                    'template':   template_img,
                    'search':     search_img,
                    'bbox':       np.array(
                        [bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=np.float32),
                    'label_cls1': labelcls1,
                    'labelxff':   labelxff,
                    'labelcls2':  labelcls2,
                    'weightxff':  weightxff,
                }

            except Exception as e:
                logger.warning(
                    f"__getitem__ attempt {attempt+1}/{MAX_RETRIES} "
                    f"failed at index {index}: {e}")
                index = np.random.randint(0, self.num)
                continue

        raise RuntimeError(
            f"TrkDataset.__getitem__: {MAX_RETRIES} consecutive failures.\n"
            f"Run tools/validate_dataset.py to diagnose missing images.\n"
            f"Last index: {index}")