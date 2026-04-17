#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/submit.py
===============
Generate submission.csv for the AIC-4 UAV Tracking Kaggle leaderboard.

Submission format (verified against sample_submission.csv):
  id,x,y,w,h
  dataset1/Car_video_0,x,y,w,h     ← frame 0 of sequence Car_video
  dataset1/Car_video_1,x,y,w,h     ← frame 1
  ...

ID format: {dataset}/{seq_name}_{frame_index}   (0-based frame index)

============================================================
BUG FIXES:
1. CRITICAL: id format verified against sample_submission.csv.
   Format is: dataset/seq_name_frameIdx (e.g. "dataset1/Car_video_0")
   NOT "dataset1/Car_video/0" or "Car_video_0"

2. CRITICAL: SEARCH_SIZE at inference must be cfg.TRAIN.SEARCH_SIZE (287)
   not cfg.TRACK.INSTANCE_SIZE (255). Using 255 gives a 9×9 feature map,
   but the hanning window and score_size are built for 11×11 (287 input).
   This mismatch causes random bbox predictions. Fixed in hift_tracker.py.

3. Annotation parsing uses universal newline mode to handle Windows \\r\\n.

4. Frame count is taken from the manifest (n_frames field) — do NOT count
   annotation lines, as some frames may have absent-target annotations.

5. If tracking fails mid-sequence, the last valid bbox is propagated forward
   (better than zeros which would score IoU=0 for those frames).

6. Added --dry-run mode: checks all paths without running the tracker.

7. All paths are absolute, resolved from this file's location.
============================================================
"""
import os
import sys
import json
import argparse
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

_THIS_DIR  = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJ_ROOT))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.hift_tracker import HiFTTracker


# ── Paths ──────────────────────────────────────────────────────────────────
MANIFEST_PATH = Path(os.environ.get(
    'CONTEST_DATA_DIR',
    str(_PROJ_ROOT.parent / 'aic4-uav-tracker' / 'data' / 'contest_release')
)).resolve() / 'metadata' / 'contestant_manifest.json'

BASE_DIR   = MANIFEST_PATH.parent.parent
OUTPUT_CSV = _PROJ_ROOT / 'submission.csv'


# ── Model loading ──────────────────────────────────────────────────────────

def load_model(model_path: str) -> ModelBuilder:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {path}')
    print(f'Loading model: {path}')
    model = ModelBuilder()
    checkpoint = torch.load(str(path), map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.cuda().eval()
    return model


# ── Annotation parsing ─────────────────────────────────────────────────────

def parse_init_bbox(anno_path: Path) -> list:
    """
    Read the FIRST line of annotation.txt with a valid [x, y, w, h] bbox.
    Returns [x, y, w, h] (top-left corner + size, 0-based).
    """
    with open(str(anno_path), 'r', newline='') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    x, y, w, h = map(float, parts[:4])
                    if w > 0 and h > 0:
                        return [x, y, w, h]
                except ValueError:
                    continue
    raise ValueError(f'No valid bbox found in {anno_path}')


# ── Tracking ───────────────────────────────────────────────────────────────

def track_sequence(model: ModelBuilder, video_path: Path,
                   init_bbox: list, n_frames: int) -> list:
    """
    Track a full sequence starting from init_bbox on frame 0.

    Returns a list of (x, y, w, h) tuples, length == n_frames.
    Frame 0 always returns init_bbox.
    Lost frames propagate the last valid prediction.
    """
    tracker = HiFTTracker(model)
    results  = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')

    ret, frame = cap.read()
    if not ret:
        raise IOError(f'Cannot read frame 0: {video_path}')

    tracker.init(frame, init_bbox)
    x, y, w, h = init_bbox
    results.append((float(x), float(y), float(w), float(h)))
    last_bbox = results[0]

    for _ in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            results.append(last_bbox)
            continue
        try:
            out = tracker.track(frame)
            bx, by, bw, bh = out['bbox']
            last_bbox = (float(bx), float(by), float(bw), float(bh))
            results.append(last_bbox)
        except Exception as e:
            results.append(last_bbox)

    cap.release()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    cfg.merge_from_file(
        str(_PROJ_ROOT / 'configs' / 'hiFT_finetune.yaml'))

    # ── Load model ────────────────────────────────────────────────────────
    if args.model:
        candidates = [args.model]
    else:
        candidates = [
            str(_PROJ_ROOT / 'snapshot' / 'best.pth'),
            str(_PROJ_ROOT / 'snapshot' / 'latest.pth'),
        ]

    model = None
    if not args.dry_run:
        for c in candidates:
            if Path(c).exists():
                model = load_model(c)
                print(f'Using checkpoint: {c}')
                break
        if model is None:
            print('ERROR: No model checkpoint found.')
            print('Train first: python tools/train.py')
            sys.exit(1)

    # ── Load manifest ─────────────────────────────────────────────────────
    if not MANIFEST_PATH.exists():
        print(f'ERROR: manifest not found: {MANIFEST_PATH}')
        print('Set CONTEST_DATA_DIR env variable.')
        sys.exit(1)

    with open(str(MANIFEST_PATH), 'r') as f:
        manifest = json.load(f)

    public_lb = manifest.get('public_lb', {})
    print(f'Tracking {len(public_lb)} sequences...\n')

    rows = []

    for seq_key, seq_info in tqdm(public_lb.items(),
                                  desc='Generating submission'):
        seq_name   = seq_info['seq_name']
        dataset    = seq_info['dataset']
        n_frames   = seq_info['n_frames']
        video_path = BASE_DIR / seq_info['video_path']
        anno_path  = BASE_DIR / seq_info['annotation_path']

        # ── Dry-run mode: just check paths ────────────────────────────────
        if args.dry_run:
            video_ok = video_path.exists()
            anno_ok  = anno_path.exists()
            status   = '✅' if (video_ok and anno_ok) else '❌'
            tqdm.write(f'{status}  {seq_key:<50} '
                       f'video={video_ok} anno={anno_ok}')
            continue

        # ── Parse init bbox ───────────────────────────────────────────────
        try:
            init_bbox = parse_init_bbox(anno_path)
        except Exception as e:
            tqdm.write(f'WARN: cannot parse init bbox for {seq_key}: {e}')
            for i in range(n_frames):
                rows.append({
                    'id': f'{dataset}/{seq_name}_{i}',
                    'x': 0, 'y': 0, 'w': 0, 'h': 0})
            continue

        # ── Track ─────────────────────────────────────────────────────────
        try:
            bboxes = track_sequence(model, video_path, init_bbox, n_frames)
        except Exception as e:
            tqdm.write(f'ERROR tracking {seq_key}: {e}')
            x, y, w, h = init_bbox
            for i in range(n_frames):
                rows.append({
                    'id': f'{dataset}/{seq_name}_{i}',
                    'x': round(x, 4), 'y': round(y, 4),
                    'w': round(w, 4), 'h': round(h, 4)})
            continue

        # ── Write rows ────────────────────────────────────────────────────
        # ID format: dataset/seq_name_frameIndex  (verified vs sample_submission.csv)
        for frame_idx, (bx, by, bw, bh) in enumerate(bboxes):
            rows.append({
                'id': f'{dataset}/{seq_name}_{frame_idx}',
                'x':  round(bx, 4),
                'y':  round(by, 4),
                'w':  round(bw, 4),
                'h':  round(bh, 4),
            })

    if args.dry_run:
        print('\nDry run complete. No submission written.')
        return

    # ── Write CSV ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=['id', 'x', 'y', 'w', 'h'])
    df.to_csv(str(OUTPUT_CSV), index=False)

    print(f'\nsubmission.csv written: {OUTPUT_CSV}')
    print(f'Total rows: {len(df):,}  (expected: 74,293)')

    # ── Sanity check against sample_submission.csv ───────────────────────
    sample_path = _PROJ_ROOT / 'sample_submission.csv'
    if sample_path.exists():
        sample_ids = set(
            pd.read_csv(str(sample_path))['id'].tolist())
        our_ids = set(df['id'].tolist())
        missing = sample_ids - our_ids
        extra   = our_ids - sample_ids
        if missing or extra:
            print(f'\n⚠️  ID MISMATCH vs sample_submission.csv!')
            print(f'   Missing: {len(missing)}  Extra: {len(extra)}')
            if missing:
                print(f'   Sample missing: {list(missing)[:5]}')
        else:
            print('\n✅ ID set matches sample_submission.csv exactly.')


def main():
    parser = argparse.ArgumentParser(
        description='Generate AIC-4 submission.csv')
    parser.add_argument('--cfg',
                        default=str(_PROJ_ROOT / 'configs' /
                                    'hiFT_finetune.yaml'))
    parser.add_argument('--model', default=None,
                        help='Path to model checkpoint (default: auto-detect)')
    parser.add_argument('--output', default=str(OUTPUT_CSV),
                        help='Output CSV path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check paths only, do not run tracker')
    args = parser.parse_args()

    global OUTPUT_CSV
    OUTPUT_CSV = Path(args.output)

    run(args)


if __name__ == '__main__':
    main()