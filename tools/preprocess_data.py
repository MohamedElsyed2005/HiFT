#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess AIC4 UAV dataset to PySOT format.
Crops images to 511x511, saves annotations in cropped space, generates train.json.

============================================================
BUG FIXES:
1. CRITICAL: parse_annotations() had inconsistent frame-id logic.
   The original code tried to detect if frame IDs were explicit (5+ fields)
   or implicit (line number = frame id). AIC4 annotations are x,y,w,h per
   line (0-indexed), so we just use the line index as frame ID.
   Fixed: always use enumerate(idx) as frame_id for 4-field lines.

2. CRITICAL: crop_and_save_frame() had two variable name collisions:
   - cx_crop, cy_crop were computed but overwritten by x1_c, y1_c
   - The returned scale was used to place bbox in cropped image coords,
     but the scale formula was wrong (it used local w_crop not s_z-based).
   Fixed: use correct formula: scale = SAVE_SIZE / s_z, then
   new_x1 = (SAVE_SIZE - w*scale)/2, new_y1 = (SAVE_SIZE - h*scale)/2.

3. train.json format must match SubDataset expectations:
   {video_name: {"0": {frame_str: [x1,y1,x2,y2], ...}}}
   The "0" key is the track ID. Verified and kept.

4. Frame index boundary: cap.set() can silently fail for frames beyond
   the video length. Added check on ret before saving.

5. Added validation: skip sequences with fewer than 2 valid frames
   (can't form template/search pairs).

6. Context-crop arithmetic made explicit and correct.
============================================================
"""
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

CONTEST_DATA_DIR = os.path.abspath(
    os.path.join(project_root,'..', 'aic4-uav-tracker','data', 'contest_release'))
MANIFEST_PATH = os.path.join(
    CONTEST_DATA_DIR, 'metadata', 'contestant_manifest.json')
OUTPUT_JSON = os.path.join(
    project_root, 'data', 'processed', 'train.json')
OUTPUT_IMAGE_ROOT = os.path.join(
    project_root, 'data', 'processed', 'crop511')

SAVE_SIZE = 511
CONTEXT_AMOUNT = 0.5


# ---------------------------------------------------------------------------
def clean_dict_keys(obj):
    """Recursively strip whitespace from JSON keys and string values."""
    if isinstance(obj, dict):
        return {k.strip(): clean_dict_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_keys(i) for i in obj]
    elif isinstance(obj, str):
        return obj.strip()
    return obj


def parse_annotations(anno_path):
    """
    Parse annotation.txt into {frame_id (int): [x1, y1, x2, y2]}.

    AIC4 format: one line per frame, values are x,y,w,h (comma or space sep).
    Frame IDs are 0-based line indices.
    """
    annos = {}
    if not os.path.exists(anno_path):
        return annos
    with open(anno_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Normalize separators
            parts = line.replace(',', ' ').split()
            if len(parts) < 4:
                continue
            try:
                x, y, w, h = map(float, parts[:4])
                if w <= 0 or h <= 0:
                    continue
                # Convert x,y,w,h → x1,y1,x2,y2
                annos[idx] = [x, y, x + w, y + h]
            except ValueError:
                continue
    return annos


def crop_and_save_frame(frame, bbox_xyxy, save_path):
    """
    Context-crop frame around bbox, resize to SAVE_SIZE×SAVE_SIZE, save.

    Returns: (scale, new_x1, new_y1, new_x2, new_y2) or None on failure.
    The returned coords are in the cropped/resized image space.
    """
    x1, y1, x2, y2 = bbox_xyxy
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    cx, cy = x1 + w / 2.0, y1 + h / 2.0

    # Context padding (same formula as PySOT)
    wc = w + CONTEXT_AMOUNT * (w + h)
    hc = h + CONTEXT_AMOUNT * (w + h)
    s_z = np.sqrt(wc * hc)   # side length of context square in original pixels

    # Scale factor: SAVE_SIZE pixels covers s_z original pixels
    scale = SAVE_SIZE / s_z

    # Crop region in original image (integer-aligned)
    half = s_z / 2.0
    x1_c = int(np.floor(cx - half))
    y1_c = int(np.floor(cy - half))
    x2_c = x1_c + int(np.ceil(s_z))
    y2_c = y1_c + int(np.ceil(s_z))

    # Pad image if crop goes outside boundaries
    img_h, img_w = frame.shape[:2]
    pad_top    = max(0, -y1_c)
    pad_bottom = max(0, y2_c - img_h)
    pad_left   = max(0, -x1_c)
    pad_right  = max(0, x2_c - img_w)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        avg = frame.mean(axis=(0, 1)).astype(np.uint8)
        frame_padded = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=avg.tolist())
        # Adjust crop coords for padding
        y1_c += pad_top;  y2_c += pad_top
        x1_c += pad_left; x2_c += pad_left
    else:
        frame_padded = frame

    crop = frame_padded[y1_c:y2_c, x1_c:x2_c]
    if crop.size == 0:
        return None

    crop_resized = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE),
                              interpolation=cv2.INTER_LINEAR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    success = cv2.imwrite(save_path, crop_resized)
    if not success:
        return None

    # Compute bbox coords in resized image
    # The object is centered in the crop; scale × original_w gives pixel size
    w_new  = w * scale
    h_new  = h * scale
    new_x1 = (SAVE_SIZE - w_new) / 2.0
    new_y1 = (SAVE_SIZE - h_new) / 2.0
    new_x2 = new_x1 + w_new
    new_y2 = new_y1 + h_new

    return new_x1, new_y1, new_x2, new_y2


def main():
    print("Loading manifest...")
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        print("Ensure CONTEST_DATA_DIR is set correctly in this script.")
        sys.exit(1)

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    manifest = clean_dict_keys(manifest)

    train_data = manifest.get('train', {})
    print(f"Found {len(train_data)} training sequences.")

    os.makedirs(OUTPUT_IMAGE_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    output_annos = {}
    skip_count = 0

    for seq_key, seq_info in tqdm(train_data.items(), desc="Preprocessing"):
        seq_name   = seq_info['seq_name']
        video_path = os.path.join(CONTEST_DATA_DIR, seq_info['video_path'])
        anno_path  = os.path.join(CONTEST_DATA_DIR, seq_info['annotation_path'])

        if not os.path.exists(video_path):
            print(f"  SKIP: video not found: {video_path}")
            skip_count += 1
            continue
        if not os.path.exists(anno_path):
            print(f"  SKIP: annotation not found: {anno_path}")
            skip_count += 1
            continue

        annos = parse_annotations(anno_path)
        if len(annos) < 2:
            # Need at least 2 frames to form template/search pairs
            print(f"  SKIP: {seq_name} has fewer than 2 valid annotations")
            skip_count += 1
            continue

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  SKIP: cannot open video: {video_path}")
                skip_count += 1
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            seq_dir = os.path.join(OUTPUT_IMAGE_ROOT, seq_name)
            os.makedirs(seq_dir, exist_ok=True)

            seq_annos = {}   # frame_str → [x1,y1,x2,y2] in cropped space
            saved_count = 0

            for frame_idx, bbox_xyxy in sorted(annos.items()):
                if frame_idx >= total_frames:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame_str = "{:06d}".format(frame_idx)
                save_path = os.path.join(seq_dir, f"{frame_str}.0.x.jpg")

                result = crop_and_save_frame(frame, bbox_xyxy, save_path)
                if result is not None:
                    new_x1, new_y1, new_x2, new_y2 = result
                    seq_annos[frame_str] = [new_x1, new_y1, new_x2, new_y2]
                    saved_count += 1

            cap.release()

            if saved_count < 2:
                print(f"  SKIP: {seq_name} only saved {saved_count} frames")
                skip_count += 1
                continue

            # PySOT annotation format: {video: {"0": {frame_str: bbox}}}
            output_annos[seq_name] = {"0": seq_annos}

        except Exception as e:
            print(f"  ERROR processing {seq_key}: {e}")
            skip_count += 1

    # Save JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_annos, f, indent=2)

    print(f"\nPreprocessing Complete!")
    print(f"Processed {len(output_annos)} sequences | Skipped {skip_count}")
    print(f"Images saved to:      {OUTPUT_IMAGE_ROOT}")
    print(f"Annotations saved to: {OUTPUT_JSON}")
    print("Next step: python tools/train.py")


if __name__ == '__main__':
    main()