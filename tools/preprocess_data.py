#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/preprocess_data.py
========================
Preprocess AIC4 UAV dataset to PySOT crop511 format WITH INTEGRATED TRAIN/VAL SPLIT.

WHAT THIS SCRIPT DOES:
1. Reads every training video + annotation from the contest release.
2. Context-crops frames around the target, resizes to SAVE_SIZE×SAVE_SIZE, saves as JPEG.
3. Builds a dataset index (manifest) keyed EXACTLY as dataset.py expects.
4. Deterministically splits manifest["train"] into 80% train / 20% val at SEQUENCE level.
5. Outputs 3 manifests:
   • manifest_full.json   (crops + public_lb metadata)
   • manifest_train.json  (80% for training)
   • manifest_val.json    (20% for validation)

CRITICAL DESIGN CHOICES:
- Directory structure matches `dataset.py` expectations exactly:
  crop511/dataset1/Car_video_2/000001.0.x.jpg
- JSON keys match the manifest's sequence identifiers (e.g. "dataset1/Car_video_2").
- Uses sequential cap.read() (NO cap.set()) to guarantee frame accuracy on H.264.
- Atomic writes prevent corrupt manifests on interruption.
- Stratified split preserves dataset distribution balance.
"""
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── Project root resolution ────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
_PROJ_ROOT   = _THIS_DIR.parent

# ── Configuration ─────────────────────────────────────────────────────────
CONTEST_DATA_DIR = Path(os.environ.get(
    'CONTEST_DATA_DIR',
    str(_PROJ_ROOT.parent / 'AIC4-UAV-Tracker' / 'data' / 'contest_release')
)).resolve()

MANIFEST_PATH      = CONTEST_DATA_DIR / 'metadata' / 'contestant_manifest.json'
OUTPUT_DIR         = _PROJ_ROOT / 'data' / 'processed'
OUTPUT_IMAGE_ROOT  = OUTPUT_DIR / 'crop511'
OUTPUT_JSON_FULL   = OUTPUT_DIR / 'manifest_full.json'
OUTPUT_JSON_TRAIN  = OUTPUT_DIR / 'manifest_train.json'
OUTPUT_JSON_VAL    = OUTPUT_DIR / 'manifest_val.json'

SAVE_SIZE          = 511
CONTEXT_AMOUNT     = 0.5
JPEG_QUALITY       = 90
MIN_VALID_FRAMES   = 2

# ── Split configuration ───────────────────────────────────────────────────
SPLIT_SEED         = 42
SPLIT_TRAIN_RATIO  = 0.8


def parse_annotations(anno_path: Path) -> dict:
    """
    Parse annotation.txt → {frame_idx (int): [x1, y1, x2, y2]}.
    Skips lines with w<=0 or h<=0 (occlusion/absence).
    """
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
    Returns: (crop_resized, nx1, ny1, nx2, ny2) in crop-space, or None.
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
    half   = s_z / 2.0
    ix1    = int(np.floor(cx - half))
    iy1    = int(np.floor(cy - half))
    side   = int(np.ceil(s_z))
    ix2    = ix1 + side
    iy2    = iy1 + side

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

    crop = frame[max(0, iy1):iy2, max(0, ix1):ix2]
    if crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
        return None

    crop_resized = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_LINEAR)

    # Map bbox to resized crop coordinates
    scale = SAVE_SIZE / s_z
    w_new = w_orig * scale
    h_new = h_orig * scale
    nx1   = (SAVE_SIZE - w_new) / 2.0
    ny1   = (SAVE_SIZE - h_new) / 2.0
    nx2   = nx1 + w_new
    ny2   = ny1 + h_new

    # Clamp
    nx1 = max(0.0, nx1); ny1 = max(0.0, ny1)
    nx2 = min(float(SAVE_SIZE), nx2); ny2 = min(float(SAVE_SIZE), ny2)

    return crop_resized, nx1, ny1, nx2, ny2


def process_sequence_sequential(seq_dir: Path, video_path: Path, annos: dict) -> dict:
    """Read video sequentially, crop & save annotated frames."""
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
                ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    save_path.write_bytes(buf.tobytes())
                    saved[frame_key] = [nx1, ny1, nx2, ny2]
        frame_idx += 1
    cap.release()
    return saved


def split_sequences_deterministic(seq_keys: list, train_ratio: float, seed: int) -> tuple:
    """Stratified sequence-level split preserving dataset distribution."""
    groups = defaultdict(list)
    for k in seq_keys:
        ds = k.split("/")[0] if "/" in k else "unknown"
        groups[ds].append(k)

    train_keys, val_keys = [], []
    rng = np.random.default_rng(seed)
    for ds in sorted(groups.keys()):
        seqs = groups[ds]
        rng.shuffle(seqs)
        idx = max(1, int(len(seqs) * train_ratio))
        train_keys.extend(seqs[:idx])
        val_keys.extend(seqs[idx:])
    return train_keys, val_keys


def _atomic_json_write(data: dict, out_path: Path):
    """Write JSON safely via temp file + rename."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(str(tmp), str(out_path))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--train-ratio', type=float, default=SPLIT_TRAIN_RATIO)
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
    public_lb  = manifest.get('public_lb', {})
    print(f'Training sequences: {len(train_data)} | Public LB: {len(public_lb)}')

    OUTPUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_annos = {}
    skip_count   = 0

    for seq_key, seq_info in tqdm(train_data.items(), desc='Preprocessing'):
        # seq_key examples: "dataset1/Car_video_2", "dataset3/car1-n"
        video_path = CONTEST_DATA_DIR / seq_info['video_path']
        anno_path  = CONTEST_DATA_DIR / seq_info['annotation_path']

        if not video_path.exists() or not anno_path.exists():
            tqdm.write(f'  SKIP (missing file): {seq_key}')
            skip_count += 1
            continue

        annos = parse_annotations(anno_path)
        if len(annos) < MIN_VALID_FRAMES:
            skip_count += 1
            continue

        # 🔑 CRITICAL: Use seq_key for directory to match dataset.py exactly
        safe_dir = seq_key.replace('/', os.sep).replace('\\', os.sep)
        seq_dir = OUTPUT_IMAGE_ROOT / safe_dir
        seq_dir.mkdir(parents=True, exist_ok=True)

        try:
            saved = process_sequence_sequential(seq_dir, video_path, annos)
        except Exception:
            skip_count += 1
            continue

        if len(saved) < MIN_VALID_FRAMES:
            skip_count += 1
            continue

        output_annos[seq_key] = {'0': saved}

    # ── Integrated Split ──────────────────────────────────────────────────
    print(f'\n[SPLIT] Performing deterministic sequence-level split...')
    all_keys = list(output_annos.keys())
    train_keys, val_keys = split_sequences_deterministic(
        all_keys, args.train_ratio, args.seed)

    print(f'[SPLIT] Train: {len(train_keys)} ({len(train_keys)/len(all_keys)*100:.1f}%)')
    print(f'[SPLIT] Val  : {len(val_keys)} ({len(val_keys)/len(all_keys)*100:.1f}%)')

    # ── Build Manifests ───────────────────────────────────────────────────
    manifest_full = {
        "train": {k: train_data[k] for k in all_keys},
        "public_lb": public_lb
    }
    manifest_train = {
        "train": {k: train_data[k] for k in train_keys},
        "public_lb": {}
    }
    manifest_val = {
        "train": {k: train_data[k] for k in val_keys},
        "public_lb": {}
    }

    _atomic_json_write(manifest_full, OUTPUT_JSON_FULL)
    _atomic_json_write(manifest_train, OUTPUT_JSON_TRAIN)
    _atomic_json_write(manifest_val, OUTPUT_JSON_VAL)

    # ── Summary ───────────────────────────────────────────────────────────
    total_frames = sum(len(v['0']) for v in output_annos.values())
    train_frames = sum(len(output_annos[k]['0']) for k in train_keys)
    val_frames   = sum(len(output_annos[k]['0']) for k in val_keys)

    print(f'\n{"="*70}')
    print(f'  PREPROCESSING COMPLETE')
    print(f'{"="*70}')
    print(f'  Processed : {len(output_annos)} | Skipped: {skip_count}')
    print(f'  Frames    : {total_frames:,} (Train: {train_frames:,} | Val: {val_frames:,})')
    print(f'  Manifests → {OUTPUT_DIR}')
    print(f'  Crops     → {OUTPUT_IMAGE_ROOT}')
    print(f'{"="*70}\n')
    print(f'Next: python tools/train.py --val-manifest {OUTPUT_JSON_VAL}')


if __name__ == '__main__':
    main()