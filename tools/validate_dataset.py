#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/validate_dataset.py
=========================
Competition-grade dataset validation script.

Scans EVERY image referenced in train.json and reports:
  - Missing files
  - Corrupt/unreadable images
  - Zero-size or degenerate bounding boxes
  - Sequences with too few valid frames
  - Overall dataset health score

Usage:
    # Quick check (metadata only, no image reads):
    python tools/validate_dataset.py --mode meta

    # Full check (reads every image with cv2.imread):
    python tools/validate_dataset.py --mode full

    # Fix mode: rebuild pick list, write cleaned JSON
    python tools/validate_dataset.py --mode fix --output data/processed/train_clean.json

Exit codes:
    0 — all good
    1 — warnings (some images missing but dataset is usable)
    2 — critical failures (>20% missing, training will be corrupted)
"""
import os
import sys
import json
import argparse
import logging
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Project root ────────────────────────────────────────────────────────────
_THIS_DIR   = Path(__file__).resolve().parent
_PROJ_ROOT  = _THIS_DIR.parent

sys.path.insert(0, str(_PROJ_ROOT))
from pysot.core.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger('validate')


# ── Helpers ──────────────────────────────────────────────────────────────────

def abs_path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_PROJ_ROOT / p).resolve()


def normalize_frame_key(k: str) -> str:
    if re.match(r'^\d+$', k):
        return '{:06d}'.format(int(k))
    return k


def check_image(path: Path) -> tuple:
    """Returns (path, ok: bool, error_msg: str)."""
    if not path.exists():
        return (path, False, 'MISSING')
    try:
        img = cv2.imread(str(path))
        if img is None:
            return (path, False, 'UNREADABLE (cv2.imread returned None)')
        if img.shape[0] < 4 or img.shape[1] < 4:
            return (path, False, f'DEGENERATE size {img.shape}')
        return (path, True, '')
    except Exception as e:
        return (path, False, f'EXCEPTION: {e}')


def check_bbox(bbox) -> tuple:
    """Returns (ok: bool, msg: str)."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) not in (2, 4):
        return False, f'bad type/length: {bbox}'
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
    else:
        w, h = bbox
    if w <= 0 or h <= 0:
        return False, f'zero/negative size: w={w} h={h}'
    return True, ''


# ── Core validation ───────────────────────────────────────────────────────────

def validate(cfg_dataset_name: str, mode: str, fix: bool, output: str):
    subdata_cfg = getattr(cfg.DATASET, cfg_dataset_name)
    root     = abs_path(subdata_cfg.ROOT)
    anno_p   = abs_path(subdata_cfg.ANNO)
    path_fmt = '{}.{}.{}.jpg'

    logger.info(f'Dataset   : {cfg_dataset_name}')
    logger.info(f'Root      : {root}')
    logger.info(f'Annotation: {anno_p}')
    logger.info(f'Mode      : {mode}')
    print()

    # ── Load JSON ─────────────────────────────────────────────────────────
    if not anno_p.exists():
        logger.error(f'Annotation JSON not found: {anno_p}')
        logger.error('Run tools/preprocess_data.py first.')
        return 2

    with open(anno_p, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    logger.info(f'JSON loaded: {len(meta)} top-level sequences')

    # ── Collect all (video, track, frame, bbox, img_path) tuples ──────────
    records = []   # (video, track, frame_key, bbox, img_path)
    bbox_errors = []

    for video, tracks in meta.items():
        video = video.replace('\\', '/')
        for trk, frames in tracks.items():
            trk = trk.replace('\\', '/')
            for frm, bbox in frames.items():
                if frm == 'frames':
                    continue
                frm_norm = normalize_frame_key(frm)
                img_path = root / video / path_fmt.format(frm_norm, trk, 'x')

                ok_bbox, bbox_msg = check_bbox(bbox)
                if not ok_bbox:
                    bbox_errors.append(
                        (video, trk, frm_norm, bbox_msg))
                    continue

                records.append((video, trk, frm_norm, bbox, img_path))

    total_frames = len(records)
    logger.info(f'Total frame records: {total_frames}  '
                f'| BBox errors skipped: {len(bbox_errors)}')

    # ── File existence / readability check ────────────────────────────────
    missing_paths  = []
    corrupt_paths  = []
    ok_count       = 0
    per_seq_status = defaultdict(lambda: {'ok': 0, 'bad': 0})

    if mode == 'meta':
        # Fast: only check existence (no cv2.imread)
        logger.info('Running metadata check (existence only)...')
        for video, trk, frm, bbox, img_path in tqdm(records, desc='Checking'):
            exists = img_path.exists()
            per_seq_status[video]['ok' if exists else 'bad'] += 1
            if exists:
                ok_count += 1
            else:
                missing_paths.append(img_path)

    else:
        # Full: read every image with cv2.imread using threads
        logger.info('Running FULL image-read check (this may take a while)...')

        def _check(rec):
            video, trk, frm, bbox, img_path = rec
            p, ok, msg = check_image(img_path)
            return video, ok, msg, img_path

        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = {ex.submit(_check, rec): rec for rec in records}
            for fut in tqdm(as_completed(futures),
                            total=len(records), desc='Reading'):
                video, ok, msg, img_path = fut.result()
                if ok:
                    ok_count += 1
                    per_seq_status[video]['ok'] += 1
                else:
                    per_seq_status[video]['bad'] += 1
                    if 'MISSING' in msg:
                        missing_paths.append(img_path)
                    else:
                        corrupt_paths.append((img_path, msg))

    # ── Per-sequence summary ───────────────────────────────────────────────
    bad_seqs = {v: s for v, s in per_seq_status.items() if s['bad'] > 0}

    print()
    print('=' * 70)
    print('  DATASET VALIDATION REPORT')
    print('=' * 70)
    print(f'  Total frame records  : {total_frames:>8,}')
    print(f'  OK (readable)        : {ok_count:>8,}  '
          f'({100*ok_count/max(total_frames,1):.1f}%)')
    print(f'  Missing files        : {len(missing_paths):>8,}  '
          f'({100*len(missing_paths)/max(total_frames,1):.1f}%)')
    print(f'  Corrupt / unreadable : {len(corrupt_paths):>8,}  '
          f'({100*len(corrupt_paths)/max(total_frames,1):.1f}%)')
    print(f'  BBox format errors   : {len(bbox_errors):>8,}')
    print(f'  Sequences with gaps  : {len(bad_seqs):>8,} / {len(per_seq_status)}')
    print('=' * 70)

    if bad_seqs:
        worst = sorted(bad_seqs.items(),
                       key=lambda x: -x[1]['bad'])[:20]
        print('\n  Worst sequences (by missing frames):')
        for seq, s in worst:
            total_seq = s['ok'] + s['bad']
            pct = 100 * s['bad'] / max(total_seq, 1)
            flag = '❌' if pct > 50 else '⚠️ '
            print(f"  {flag}  {seq:<50}  "
                  f"missing {s['bad']:>4}/{total_seq} ({pct:5.1f}%)")

    if missing_paths[:5]:
        print(f'\n  Sample missing paths (first 5):')
        for p in missing_paths[:5]:
            print(f'    {p}')

    # ── Fix mode: write cleaned JSON ──────────────────────────────────────
    if fix and output:
        logger.info(f'Fix mode: writing cleaned JSON to {output}')
        # Build set of valid (video/trk/frm) keys
        valid_set = set()
        for video, trk, frm, bbox, img_path in records:
            if img_path.exists():
                valid_set.add((video, trk, frm))

        clean_meta = {}
        for video, tracks in meta.items():
            video_n = video.replace('\\', '/')
            new_tracks = {}
            for trk, frames in tracks.items():
                trk_n = trk.replace('\\', '/')
                new_frames = {}
                for frm, bbox in frames.items():
                    if frm == 'frames':
                        continue
                    frm_norm = normalize_frame_key(frm)
                    if (video_n, trk_n, frm_norm) in valid_set:
                        new_frames[frm_norm] = bbox
                if len(new_frames) >= 2:
                    new_tracks[trk_n] = new_frames
            if new_tracks:
                clean_meta[video_n] = new_tracks

        out_path = abs_path(output) if output else None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(clean_meta, f, indent=2)

        removed_seqs   = len(meta) - len(clean_meta)
        removed_frames = total_frames - ok_count
        print(f'\n  Clean JSON written: {out_path}')
        print(f'  Removed sequences: {removed_seqs}')
        print(f'  Removed frames   : {removed_frames}')

    # ── Exit code ─────────────────────────────────────────────────────────
    bad_total = len(missing_paths) + len(corrupt_paths)
    pct_bad = bad_total / max(total_frames, 1)

    if pct_bad == 0:
        print('\n  ✅ Dataset is CLEAN — ready for training.')
        return 0
    elif pct_bad < 0.20:
        print(f'\n  ⚠️  {pct_bad*100:.1f}% of frames are bad — '
              f'dataset is usable but consider re-preprocessing.')
        return 1
    else:
        print(f'\n  ❌ CRITICAL: {pct_bad*100:.1f}% of frames missing/corrupt. '
              f'Re-run tools/preprocess_data.py before training.')
        return 2


# ── Additional: check for the data directory structure ────────────────────

def check_directory_structure():
    """Quick sanity check on the expected directory layout."""
    print('\n── Directory Structure Check ──')
    expected = [
        _PROJ_ROOT / 'data' / 'processed' / 'crop511',
        _PROJ_ROOT / 'data' / 'processed' / 'train.json',
        _PROJ_ROOT / 'configs' / 'hiFT_finetune.yaml',
        _PROJ_ROOT / 'pretrained_models' / 'first.pth',
    ]
    for p in expected:
        exists = p.exists()
        icon = '✅' if exists else '❌'
        print(f'  {icon}  {p.relative_to(_PROJ_ROOT)}')
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Validate AIC4 training dataset pipeline')
    parser.add_argument('--cfg',
                        default=str(_PROJ_ROOT / 'configs' /
                                    'hiFT_finetune.yaml'),
                        help='Path to YAML config')
    parser.add_argument('--mode', choices=['meta', 'full'],
                        default='meta',
                        help='meta=existence only, full=cv2.imread each file')
    parser.add_argument('--fix', action='store_true',
                        help='Write a cleaned JSON with only valid frames')
    parser.add_argument('--output',
                        default='data/processed/train_clean.json',
                        help='Output path for cleaned JSON (used with --fix)')
    parser.add_argument('--dataset', default=None,
                        help='Override dataset name (default: from config)')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)

    check_directory_structure()

    dataset_names = [args.dataset] if args.dataset else list(cfg.DATASET.NAMES)
    worst_code = 0

    for ds_name in dataset_names:
        print(f'\n{"="*70}')
        print(f'  Validating: {ds_name}')
        print(f'{"="*70}')
        code = validate(ds_name, args.mode, args.fix, args.output)
        worst_code = max(worst_code, code)

    print(f'\nFinal exit code: {worst_code}')
    sys.exit(worst_code)


if __name__ == '__main__':
    main()