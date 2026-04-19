#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/preprocess_data.py  (FIXED)
==================================
CRITICAL BUG FIXED:
  The original script computed 'output_annos' (the dict of crop-space bounding
  boxes keyed by seq/track/frame) but NEVER wrote it to disk as train.json /
  val.json.  Instead it wrote the raw manifest metadata slices to
  manifest_train.json / manifest_val.json — a completely different format that
  dataset.py cannot read.

  dataset.py expects ANNO to be a JSON file of the form:
      {
        "dataset1/Car_video_2": {
          "0": {
            "000001": [x1, y1, x2, y2],
            ...
          }
        },
        ...
      }

  This script now writes exactly that structure to:
      data/processed/train.json    (80% split — used by cfg.DATASET.AIC4.ANNO)
      data/processed/val.json      (20% split — used by --val-manifest)
      data/processed/manifest_full.json  (all sequences, for reference)

  The raw manifest metadata slices (video_path, annotation_path etc.) are no
  longer written — they are not needed by the training pipeline.

UNCHANGED logic:
  - make_crop() (context-pad → resize to SAVE_SIZE=511 → store corner bbox)
  - sequential cap.read() for frame accuracy
  - stratified 80/20 sequence-level split
  - atomic JSON writes
  - directory structure: crop511/{seq_key}/{frame:06d}.0.x.jpg
"""
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── Project root ─────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
_PROJ_ROOT   = _THIS_DIR.parent

# ── Configuration ─────────────────────────────────────────────────────────────
CONTEST_DATA_DIR = Path(os.environ.get(
    'CONTEST_DATA_DIR',
    str(_PROJ_ROOT.parent / 'AIC4-UAV-Tracker' / 'data' / 'contest_release')
)).resolve()

MANIFEST_PATH     = CONTEST_DATA_DIR / 'metadata' / 'contestant_manifest.json'
OUTPUT_DIR        = _PROJ_ROOT / 'data' / 'processed'
OUTPUT_IMAGE_ROOT = OUTPUT_DIR / 'crop511'

# ─────────────────────────────────────────────────────────────────────────────
# FIXED output paths — these are what cfg.DATASET.AIC4.ANNO points to
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_JSON_TRAIN = OUTPUT_DIR / 'train.json'    # ← cfg.DATASET.AIC4.ANNO
OUTPUT_JSON_VAL   = OUTPUT_DIR / 'val.json'      # ← --val-manifest
OUTPUT_JSON_FULL  = OUTPUT_DIR / 'manifest_full.json'

SAVE_SIZE         = 511
CONTEXT_AMOUNT    = 0.5
JPEG_QUALITY      = 90
MIN_VALID_FRAMES  = 2

SPLIT_SEED        = 42
SPLIT_TRAIN_RATIO = 0.8


def parse_annotations(anno_path: Path) -> dict:
    """Parse annotation.txt → {frame_idx (int): [x1, y1, x2, y2]}."""
    annos = {}
    if not anno_path.exists():
        return annos
    with open(anno_path, 'r', newline='') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 4:
                continue
            try:
                x, y, w, h = map(float, parts[:4])
            except ValueError:
                continue
            if w > 0 and h > 0:
                annos[idx] = [x, y, x + w, y + h]
    return annos


def make_crop(frame: np.ndarray, bbox_xyxy: list):
    """
    Context-crop frame around bbox, pad borders, resize to SAVE_SIZE.
    Returns (crop_resized, nx1, ny1, nx2, ny2) in crop-space, or None.
    The target is always centred in the output crop.
    """
    x1, y1, x2, y2 = bbox_xyxy
    w_orig, h_orig = x2 - x1, y2 - y1
    if w_orig <= 0 or h_orig <= 0:
        return None

    cx = x1 + w_orig / 2.0
    cy = y1 + h_orig / 2.0

    wc  = w_orig + CONTEXT_AMOUNT * (w_orig + h_orig)
    hc  = h_orig + CONTEXT_AMOUNT * (w_orig + h_orig)
    s_z = np.sqrt(wc * hc)

    half = s_z / 2.0
    ix1  = int(np.floor(cx - half))
    iy1  = int(np.floor(cy - half))
    side = int(np.ceil(s_z))
    ix2  = ix1 + side
    iy2  = iy1 + side

    img_h, img_w = frame.shape[:2]

    pad_top    = max(0, -iy1)
    pad_bottom = max(0, iy2 - img_h)
    pad_left   = max(0, -ix1)
    pad_right  = max(0, ix2 - img_w)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        avg = np.mean(frame, axis=(0, 1)).astype(np.uint8).tolist()
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=avg)
        iy1 += pad_top;  iy2 += pad_top
        ix1 += pad_left; ix2 += pad_left

    crop = frame[iy1:iy2, ix1:ix2]

    if crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
        return None

    crop_resized = cv2.resize(
        crop, (SAVE_SIZE, SAVE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # ===== Correct bbox mapping =====
    scale = SAVE_SIZE / s_z
    w_new = w_orig * scale
    h_new = h_orig * scale

    nx1 = (SAVE_SIZE - w_new) / 2.0
    ny1 = (SAVE_SIZE - h_new) / 2.0
    nx2 = nx1 + w_new
    ny2 = ny1 + h_new

    # ===== Clamp =====
    nx1 = max(0.0, nx1)
    ny1 = max(0.0, ny1)
    nx2 = min(float(SAVE_SIZE), nx2)
    ny2 = min(float(SAVE_SIZE), ny2)

    # ===== VALIDATION =====
    if nx1 >= nx2 or ny1 >= ny2:
        return None

    min_size = 2.0
    if (nx2 - nx1) < min_size or (ny2 - ny1) < min_size:
        return None

    return crop_resized, nx1, ny1, nx2, ny2


def process_sequence_sequential(seq_dir: Path, video_path: Path,
                                 annos: dict) -> dict:
    """Read video sequentially (NO cap.set()), crop and save annotated frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    saved         = {}
    frame_idx     = 0
    annotated_set = set(annos.keys())
    max_frame     = max(annotated_set) if annotated_set else -1

    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in annotated_set:
            result = make_crop(frame, annos[frame_idx])
            if result is not None:
                crop, nx1, ny1, nx2, ny2 = result
                frame_key = '{:06d}'.format(frame_idx)
                save_path = seq_dir / f'{frame_key}.0.x.jpg'
                ok, buf = cv2.imencode(
                    '.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    save_path.write_bytes(buf.tobytes())
                    saved[frame_key] = [nx1, ny1, nx2, ny2]
        frame_idx += 1

    cap.release()
    return saved


def split_sequences_deterministic(seq_keys: list, train_ratio: float,
                                   seed: int, annos=None) -> tuple:
    """Stratified sequence-level split preserving dataset + length distribution."""
    from collections import defaultdict
    import numpy as np

    def get_length_bucket(n_frames):
        if n_frames < 100:
            return 'short'
        elif n_frames < 300:
            return 'medium'
        else:
            return 'long'

    groups = defaultdict(list)

    for k in seq_keys:
        ds = k.split("/")[0] if "/" in k else "unknown"

        # ===== length-aware stratification =====
        if annos is not None and k in annos:
            try:
                obj = next(iter(annos[k].values()))
                n_frames = len(obj.get('frames', []))
            except Exception:
                n_frames = 0
        else:
            n_frames = 0

        length_bucket = get_length_bucket(n_frames)

        key = f"{ds}_{length_bucket}"
        groups[key].append(k)

    train_keys, val_keys = [], []
    rng = np.random.default_rng(seed)

    for key in sorted(groups.keys()):
        seqs = groups[key]
        rng.shuffle(seqs)

        if len(seqs) == 0:
            continue

        idx = max(1, int(len(seqs) * train_ratio))

        train_keys.extend(seqs[:idx])
        val_keys.extend(seqs[idx:])

    return train_keys, val_keys


def _atomic_json_write(data: dict, out_path: Path):
    """Write JSON safely via temp file + atomic rename."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(str(tmp), str(out_path))
    print(f'  Written: {out_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        type=int,   default=SPLIT_SEED)
    parser.add_argument('--train-ratio', type=float, default=SPLIT_TRAIN_RATIO)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip seq directories that already have crops')
    args = parser.parse_args()

    print(f'Contest data : {CONTEST_DATA_DIR}')
    print(f'Output dir   : {OUTPUT_DIR}')
    print(f'Split seed   : {args.seed} | Ratio: {args.train_ratio}')
    print()

    if not MANIFEST_PATH.exists():
        sys.exit(f'ERROR: manifest not found: {MANIFEST_PATH}')

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    train_data = manifest.get('train', {})
    print(f'Training sequences in manifest: {len(train_data)}')

    OUTPUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── output_annos: the crop-annotation dict that dataset.py reads ─────────
    # Structure: {seq_key: {"0": {frame_key: [x1,y1,x2,y2], ...}}}
    output_annos = {}
    skip_count   = 0

    for seq_key, seq_info in tqdm(train_data.items(), desc='Preprocessing'):
        video_path = CONTEST_DATA_DIR / seq_info['video_path']
        anno_path  = CONTEST_DATA_DIR / seq_info['annotation_path']

        if not video_path.exists() or not anno_path.exists():
            tqdm.write(f'  SKIP (missing): {seq_key}')
            skip_count += 1
            continue

        annos = parse_annotations(anno_path)
        if len(annos) < MIN_VALID_FRAMES:
            skip_count += 1
            continue

        safe_dir = seq_key.replace('/', os.sep).replace('\\', os.sep)
        seq_dir  = OUTPUT_IMAGE_ROOT / safe_dir
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Optional: skip re-processing if images already exist
        if args.skip_existing:
            existing = list(seq_dir.glob('*.x.jpg'))
            if len(existing) >= MIN_VALID_FRAMES:
                # Rebuild annotation dict from existing files
                frame_anno = {}
                for img_p in existing:
                    frame_key = img_p.stem.split('.')[0]
                    frame_int = int(frame_key)
                    if frame_int in annos:
                        x1, y1, x2, y2 = annos[frame_int]
                        w_orig = x2 - x1
                        h_orig = y2 - y1
                        if w_orig > 0 and h_orig > 0:
                            wc = w_orig + CONTEXT_AMOUNT * (w_orig + h_orig)
                            hc = h_orig + CONTEXT_AMOUNT * (w_orig + h_orig)
                            s_z = np.sqrt(wc * hc)
                            scale = SAVE_SIZE / s_z
                            w_new = w_orig * scale
                            h_new = h_orig * scale
                            nx1 = (SAVE_SIZE - w_new) / 2.0
                            ny1 = (SAVE_SIZE - h_new) / 2.0
                            frame_anno[frame_key] = [
                                max(0.0, nx1), max(0.0, ny1),
                                min(float(SAVE_SIZE), nx1 + w_new),
                                min(float(SAVE_SIZE), ny1 + h_new)
                            ]
                if len(frame_anno) >= MIN_VALID_FRAMES:
                    output_annos[seq_key] = {'0': frame_anno}
                    continue

        try:
            saved = process_sequence_sequential(seq_dir, video_path, annos)
        except Exception as e:
            tqdm.write(f'  ERROR {seq_key}: {e}')
            skip_count += 1
            continue

        if len(saved) < MIN_VALID_FRAMES:
            skip_count += 1
            continue

        # ─────────────────────────────────────────────────────────────────────
        # Store crop annotations keyed as dataset.py expects:
        #   output_annos[seq_key]["0"][frame_key] = [x1, y1, x2, y2]
        # ─────────────────────────────────────────────────────────────────────
        output_annos[seq_key] = {'0': saved}

    print(f'\nProcessed: {len(output_annos)}  Skipped: {skip_count}')

    if len(output_annos) == 0:
        sys.exit('ERROR: no sequences processed. Check CONTEST_DATA_DIR.')

    # ── Split ─────────────────────────────────────────────────────────────────
    all_keys = list(output_annos.keys())
    train_keys, val_keys = split_sequences_deterministic(
        all_keys, args.train_ratio, args.seed)

    print(f'\n[SPLIT] Train: {len(train_keys)}  Val: {len(val_keys)}')

    train_annos = {k: output_annos[k] for k in train_keys}
    val_annos   = {k: output_annos[k] for k in val_keys}

    # ── Write JSON files ──────────────────────────────────────────────────────
    # train.json and val.json are the CROP ANNOTATION files (what dataset.py reads).
    # They have the structure dataset.py expects: {seq_key: {"0": {frame: bbox}}}
    print('\nWriting annotation JSONs...')
    _atomic_json_write(output_annos, OUTPUT_JSON_FULL)
    _atomic_json_write(train_annos,  OUTPUT_JSON_TRAIN)   # cfg points here
    _atomic_json_write(val_annos,    OUTPUT_JSON_VAL)

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_frames = sum(len(v['0']) for v in output_annos.values())
    train_frames = sum(len(output_annos[k]['0']) for k in train_keys)
    val_frames   = sum(len(output_annos[k]['0']) for k in val_keys)

    print(f'\n{"="*70}')
    print(f'  PREPROCESSING COMPLETE')
    print(f'{"="*70}')
    print(f'  Sequences : {len(output_annos)} total '
          f'(Train: {len(train_keys)} | Val: {len(val_keys)})')
    print(f'  Frames    : {total_frames:,} '
          f'(Train: {train_frames:,} | Val: {val_frames:,})')
    print(f'  Crops     → {OUTPUT_IMAGE_ROOT}')
    print(f'  train.json→ {OUTPUT_JSON_TRAIN}')
    print(f'  val.json  → {OUTPUT_JSON_VAL}')
    print(f'{"="*70}\n')
    print(f'Verify with:')
    print(f'  VALIDATE_ON_INIT=1 python tools/train.py')
    print(f'  python tools/validate_dataset.py --mode meta')


if __name__ == '__main__':
    main()