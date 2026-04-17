#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script: computes AUC (Success Rate) and Precision on public_lb.

Usage:
    python tools/eval.py [--model snapshot/best.pth]

Metrics:
  - Success Rate (AUC): fraction of frames where IoU(pred, gt) > threshold,
    averaged over thresholds [0, 0.05, 0.10, ..., 1.0]
  - Precision: fraction of frames where center error < threshold (20px default)
  - Normalized Precision: same but scaled by target size
"""
import os
import sys
import json
import cv2
import argparse
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.hift_tracker import HiFTTracker


MANIFEST_PATH = os.path.join(
    PROJECT_ROOT, 'data', 'contest_release', 'metadata',
    'contestant_manifest.json')
BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'contest_release')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=os.path.join(
        PROJECT_ROOT, 'configs', 'hiFT_finetune.yaml'))
    parser.add_argument('--model', default=None,
                        help='path to model checkpoint')
    return parser.parse_args()


def load_model(path):
    model = ModelBuilder()
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.cuda().eval()
    return model


def load_gt(anno_path):
    """Load all GT bboxes [x,y,w,h] per frame."""
    bboxes = []
    with open(anno_path, 'r', newline='') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    x, y, w, h = map(float, parts[:4])
                    bboxes.append([x, y, w, h])
                except ValueError:
                    bboxes.append([0, 0, 0, 0])
    return bboxes


def iou_xywh(b1, b2):
    """IoU between two [x,y,w,h] boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / (union + 1e-6)


def center_error(b1, b2):
    """Euclidean distance between centers."""
    cx1, cy1 = b1[0] + b1[2]/2, b1[1] + b1[3]/2
    cx2, cy2 = b2[0] + b2[2]/2, b2[1] + b2[3]/2
    return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)


def evaluate_sequence(model, video_path, gt_bboxes):
    """Track sequence and return per-frame IoU and center error."""
    tracker = HiFTTracker(model)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    ious  = []
    cerrs = []
    init_done = False
    last_pred = None

    for frame_idx, gt in enumerate(gt_bboxes):
        ret, frame = cap.read()
        if not ret:
            break

        if not init_done:
            tracker.init(frame, gt)
            init_done = True
            last_pred = gt
            ious.append(1.0)
            cerrs.append(0.0)
            continue

        try:
            out = tracker.track(frame)
            pred = out['bbox']  # [x,y,w,h]
            last_pred = pred
        except Exception:
            pred = last_pred

        ious.append(iou_xywh(pred, gt))
        cerrs.append(center_error(pred, gt))

    cap.release()
    return np.array(ious), np.array(cerrs)


def compute_auc(ious, thresholds=None):
    """AUC of success curve over IoU thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)
    success = [np.mean(ious >= t) for t in thresholds]
    return np.mean(success)


def compute_precision(cerrs, threshold=20):
    """Fraction of frames with center error < threshold pixels."""
    return np.mean(cerrs < threshold)


def main():
    import torch  # import here so eval.py can be imported without torch

    args = parse_args()
    cfg.merge_from_file(args.cfg)

    model_path = args.model
    if model_path is None:
        for candidate in ['snapshot/best.pth', 'snapshot/latest.pth']:
            p = os.path.join(PROJECT_ROOT, candidate)
            if os.path.exists(p):
                model_path = p
                break

    if model_path is None or not os.path.exists(model_path):
        print("ERROR: No model found. Train first or specify --model.")
        sys.exit(1)

    print(f"Evaluating model: {model_path}")
    model = load_model(model_path)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    public_lb = manifest.get('public_lb', {})
    print(f"Evaluating on {len(public_lb)} sequences...")

    all_ious  = []
    all_cerrs = []
    per_seq   = []

    for seq_key, seq_info in tqdm(public_lb.items()):
        video_path = os.path.join(BASE_DIR, seq_info['video_path'])
        anno_path  = os.path.join(BASE_DIR, seq_info['annotation_path'])

        if not os.path.exists(video_path) or not os.path.exists(anno_path):
            continue

        gt = load_gt(anno_path)
        if len(gt) < 2:
            continue

        ious, cerrs = evaluate_sequence(model, video_path, gt)
        if ious is None:
            continue

        seq_auc  = compute_auc(ious)
        seq_prec = compute_precision(cerrs)
        per_seq.append({
            'seq': seq_key,
            'auc': seq_auc,
            'precision': seq_prec,
            'n_frames': len(ious)
        })
        all_ious.extend(ious.tolist())
        all_cerrs.extend(cerrs.tolist())

    if not all_ious:
        print("No sequences evaluated.")
        return

    all_ious  = np.array(all_ious)
    all_cerrs = np.array(all_cerrs)

    overall_auc  = compute_auc(all_ious)
    overall_prec = compute_precision(all_cerrs)

    print(f"\n{'='*50}")
    print(f"Overall AUC       : {overall_auc:.4f}")
    print(f"Overall Precision : {overall_prec:.4f}")
    print(f"(at 20px threshold)")
    print(f"Evaluated {len(per_seq)} sequences / {len(all_ious)} frames")
    print(f"{'='*50}")

    # Top-5 worst sequences (to identify failure modes)
    per_seq.sort(key=lambda x: x['auc'])
    print("\nWorst 5 sequences (lowest AUC):")
    for s in per_seq[:5]:
        print(f"  {s['seq']:<40} AUC={s['auc']:.3f}  Prec={s['precision']:.3f}")


if __name__ == '__main__':
    import torch
    main()