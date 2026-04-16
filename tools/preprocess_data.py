#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess AIC4 UAV dataset to PySOT format.
Crops images to 511x511, saves annotations in cropped space, and generates train.json.
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

# Adjust this path if your dataset is located elsewhere
CONTEST_DATA_DIR = os.path.abspath(os.path.join(project_root, '..', 'aic4-uav-tracker', 'data', 'contest_release'))
MANIFEST_PATH = os.path.join(CONTEST_DATA_DIR, 'metadata', 'contestant_manifest.json')
OUTPUT_JSON = os.path.join(project_root, 'data', 'processed', 'train.json')
OUTPUT_IMAGE_ROOT = os.path.join(project_root, 'data', 'processed', 'crop511')

SAVE_SIZE = 511
CONTEXT_AMOUNT = 0.5
# ---------------------------------------------------------------------------

def clean_dict_keys(obj):
    """Recursively strip trailing/leading whitespace from JSON keys and string values"""
    if isinstance(obj, dict):
        return {k.strip(): clean_dict_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_keys(i) for i in obj]
    elif isinstance(obj, str):
        return obj.strip()
    return obj

def parse_annotations(anno_path):
    """Parse annotation.txt into a dictionary {frame_id: [x1, y1, x2, y2]}"""
    annos = {}
    if not os.path.exists(anno_path):
        return annos
    with open(anno_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().replace(',', ' ').split()
            if len(parts) < 4: continue
            try:
                # Detect if frame_id is explicitly provided or if lines are just consecutive coordinates
                if len(parts) >= 5:
                    frame_id = int(float(parts[0]))
                    x1, y1, x2, y2 = map(float, parts[1:5])
                else:
                    frame_id = idx
                    x1, y1, w, h = map(float, parts[:4])
                    x2, y2 = x1 + w, y1 + h
                
                # Convert to 0-based indexing if necessary
                if frame_id > 100 and min(annos.keys()) == 1 if annos else False:
                    frame_id -= 1
                    
                annos[frame_id] = [x1, y1, x2, y2]
            except ValueError:
                continue
    return annos

def crop_and_save_frame(frame, bbox, save_path):
    """
    Crop frame around bbox, resize to SAVE_SIZE, save to disk.
    Returns: (True, scale, orig_w, orig_h) on success, False on failure.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False

    cx, cy = x1 + w/2, y1 + h/2
    wc = w + CONTEXT_AMOUNT * (w + h)
    hc = h + CONTEXT_AMOUNT * (w + h)
    s_z = np.sqrt(wc * hc)
    scale = SAVE_SIZE / s_z

    w_crop, h_crop = w * scale, h * scale
    cx_crop, cy_crop = cx - w_crop/2, cy - h_crop/2
    x1_c, y1_c = cx + w_crop/2, cy + h_crop/2
    
    x1_c = max(0, int(cx - w_crop/2))
    y1_c = max(0, int(cy - h_crop/2))
    x2_c = min(frame.shape[1], int(cx + w_crop/2))
    y2_c = min(frame.shape[0], int(cy + h_crop/2))

    crop = frame[y1_c:y2_c, x1_c:x2_c]
    if crop.size == 0:
        return False

    crop_resized = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_LINEAR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, crop_resized)
    return True, scale, w, h

def main():
    print("📂 Loading manifest...")
    if not os.path.exists(MANIFEST_PATH):
        print(f"❌ Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # Fix whitespace issues in JSON keys/values
    manifest = clean_dict_keys(manifest)
    train_data = manifest.get('train', {})
    print(f"✅ Found {len(train_data)} training sequences.")

    os.makedirs(OUTPUT_IMAGE_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    output_annos = {}
    skip_count = 0

    for seq_key, seq_info in tqdm(train_data.items(), desc="Processing"):
        seq_name = seq_info['seq_name']
        video_path = os.path.join(CONTEST_DATA_DIR, seq_info['video_path'])
        anno_path = os.path.join(CONTEST_DATA_DIR, seq_info['annotation_path'])

        if not os.path.exists(video_path) or not os.path.exists(anno_path):
            skip_count += 1
            continue

        annos = parse_annotations(anno_path)
        if not annos:
            skip_count += 1
            continue

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                skip_count += 1
                continue

            seq_dir = os.path.join(OUTPUT_IMAGE_ROOT, seq_name)
            os.makedirs(seq_dir, exist_ok=True)
            output_annos[seq_name] = {"0": {}}
            saved_count = 0

            for frame_idx, bbox in annos.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: continue

                frame_str = f"{frame_idx:06d}"
                save_path = os.path.join(seq_dir, f"{frame_str}.0.x.jpg")
                
                res = crop_and_save_frame(frame, bbox, save_path)
                if res:
                    _, scale, w, h = res
                    # Calculate bbox coordinates relative to the 511x511 cropped image
                    w_new, h_new = w * scale, h * scale
                    new_x1 = (SAVE_SIZE - w_new) / 2
                    new_y1 = (SAVE_SIZE - h_new) / 2
                    new_x2 = new_x1 + w_new
                    new_y2 = new_y1 + h_new
                    
                    output_annos[seq_name]["0"][frame_str] = [new_x1, new_y1, new_x2, new_y2]
                    saved_count += 1
            cap.release()

            if saved_count == 0:
                del output_annos[seq_name]
                skip_count += 1
                
        except Exception as e:
            print(f"\n⚠️ Error processing {seq_key}: {e}")
            skip_count += 1

    # Save JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_annos, f, indent=2)

    print(f"\n✅ Preprocessing Complete!")
    print(f"📊 Processed {len(output_annos)} sequences | Skipped {skip_count}")
    print(f"📁 Images saved to: {OUTPUT_IMAGE_ROOT}")
    print(f"📄 Annotations saved to: {OUTPUT_JSON}")
    print("👉 Next step: python tools/train.py")

if __name__ == '__main__':
    main()