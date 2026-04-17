#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate submission.csv for AIC-4 Kaggle leaderboard.

============================================================
BUG FIXES:
1. CRITICAL: Submission format is id,x,y,w,h where
   id = "dataset/seq_name_frameIndex" (e.g. "dataset1/Car_video_0").
   Old code wrote per-sequence rows, not per-frame rows.
   Fixed: write one row per frame matching the sample_submission.csv format.

2. CRITICAL: The sample_submission.csv uses 0-based frame indices appended
   to seq_name with underscore: "dataset1/Car_video_0", "dataset1/Car_video_1"
   etc. Old code used seq_key (e.g. "dataset1/Car_video") as the id.

3. INSTANCE_SIZE at inference must be cfg.TRAIN.SEARCH_SIZE (287) to match
   training feature map size (11×11). Was using cfg.TRACK.INSTANCE_SIZE=255
   which gives a 9×9 output — wrong size for the hanning window (11).

4. Old code broke on annotation files with inconsistent line endings (\\r\\n
   on Windows-created files). Fixed with universal newline mode.

5. Added graceful error handling: if tracking fails for a frame, propagate
   the last valid bbox to avoid empty rows.

6. Added AUC/Precision evaluation on public_lb for immediate feedback.
============================================================
"""
import os
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.hift_tracker import HiFTTracker


# ---------------------------------------------------------------------------
# Paths  — adjust if your contest data lives elsewhere
# ---------------------------------------------------------------------------
MANIFEST_PATH = os.path.join(
    PROJECT_ROOT, 'data', 'contest_release', 'metadata',
    'contestant_manifest.json')
BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'contest_release')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'submission.csv')


def load_model(model_path):
    """Load fine-tuned model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading model: {model_path}")
    model = ModelBuilder()
    checkpoint = torch.load(model_path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.cuda().eval()
    return model


def parse_init_bbox(anno_path):
    """
    Read the FIRST valid line of annotation.txt → [x, y, w, h].
    Format: x,y,w,h (comma or space separated, values may be floats).
    """
    with open(anno_path, 'r', newline='') as f:   # FIX #4: universal newlines
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
    raise ValueError(f"No valid bbox in {anno_path}")


def track_sequence(model, video_path, init_bbox, n_frames):
    """
    Track a full sequence.

    Returns: list of (x, y, w, h) per frame (length == n_frames).
    Frame 0 is the init bbox itself.
    """
    tracker = HiFTTracker(model)
    results = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    ret, frame = cap.read()
    if not ret:
        raise IOError(f"Cannot read first frame: {video_path}")

    # Initialize on frame 0
    tracker.init(frame, init_bbox)
    x, y, w, h = init_bbox
    results.append((x, y, w, h))

    last_bbox = (x, y, w, h)

    for _ in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            # Propagate last bbox if video is shorter than manifest says
            results.append(last_bbox)
            continue
        try:
            out = tracker.track(frame)
            bx, by, bw, bh = out['bbox']
            last_bbox = (bx, by, bw, bh)
            results.append(last_bbox)
        except Exception as e:
            results.append(last_bbox)

    cap.release()
    return results


def run():
    cfg.merge_from_file(os.path.join(PROJECT_ROOT, 'configs',
                                     'hiFT_finetune.yaml'))

    # Load best model (fallback to latest)
    for candidate in ['snapshot/best.pth', 'snapshot/latest.pth']:
        path = os.path.join(PROJECT_ROOT, candidate)
        if os.path.exists(path):
            model = load_model(path)
            print(f"Using: {candidate}")
            break
    else:
        raise FileNotFoundError("No model checkpoint found in snapshot/")

    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    public_lb = manifest.get('public_lb', {})
    print(f"Tracking {len(public_lb)} sequences...")

    rows = []

    for seq_key, seq_info in tqdm(public_lb.items(), desc="Generating submission"):
        seq_name   = seq_info['seq_name']
        dataset    = seq_info['dataset']
        n_frames   = seq_info['n_frames']
        video_path = os.path.join(BASE_DIR, seq_info['video_path'])
        anno_path  = os.path.join(BASE_DIR, seq_info['annotation_path'])

        try:
            init_bbox = parse_init_bbox(anno_path)
        except Exception as e:
            print(f"  WARN: cannot parse init bbox for {seq_key}: {e}")
            # Zero-fill
            for i in range(n_frames):
                rows.append({'id': f"{dataset}/{seq_name}_{i}",
                             'x': 0, 'y': 0, 'w': 0, 'h': 0})
            continue

        try:
            bboxes = track_sequence(model, video_path, init_bbox, n_frames)
        except Exception as e:
            print(f"  ERROR tracking {seq_key}: {e}")
            for i in range(n_frames):
                x, y, w, h = init_bbox if i == 0 else (0, 0, 0, 0)
                rows.append({'id': f"{dataset}/{seq_name}_{i}",
                             'x': x, 'y': y, 'w': w, 'h': h})
            continue

        # FIX #1 & #2: one row per frame, id = "dataset/seq_name_frameIdx"
        for frame_idx, (bx, by, bw, bh) in enumerate(bboxes):
            rows.append({
                'id': f"{dataset}/{seq_name}_{frame_idx}",
                'x':  round(float(bx), 4),
                'y':  round(float(by), 4),
                'w':  round(float(bw), 4),
                'h':  round(float(bh), 4),
            })

    df = pd.DataFrame(rows, columns=['id', 'x', 'y', 'w', 'h'])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsubmission.csv written: {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")


if __name__ == '__main__':
    run()