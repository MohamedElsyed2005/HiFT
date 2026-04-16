#!/usr/bin/env python
import os
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.hift_tracker import HiFTTracker

def load_model(model_path):
    """Loads the model from the specified checkpoint path."""
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Did training finish?")
        
    model = ModelBuilder()
    checkpoint = torch.load(model_path, map_location='cuda')
    # strict=False handles any minor key mismatches
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda().eval()
    return model

def run():
    # Load config
    cfg.merge_from_file('configs/hiFT_finetune.yaml')
    
    # Load the BEST model
    model_path = 'snapshot/best.pth' 
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        # Fallback to latest if best doesn't exist
        print("Warning: best.pth not found. Trying latest.pth...")
        model = load_model('snapshot/latest.pth')
    
    manifest_path = '../AIC-4/aic4-uav-tracker/data/contest_release/metadata/contestant_manifest.json'
    base_dir = os.path.abspath('../AIC-4/aic4-uav-tracker/data/contest_release')
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    public_lb = manifest.get('public_lb', {})
    results = []
    
    for seq_key, seq_info in tqdm(public_lb.items(), desc="Tracking Public LB"):
        video_path = os.path.join(base_dir, seq_info['video_path'])
        anno_path = os.path.join(base_dir, seq_info['annotation_path'])
        
        # Read init bbox from annotation (first frame)
        # Assuming format: x,y,w,h (based on previous preprocessing)
        with open(anno_path, 'r') as f:
            line = f.readline().strip().replace(',', ' ')
            parts = list(map(float, line.split()[:4]))
            if len(parts) == 4:
                x, y, w, h = parts
            else:
                # Fallback if format is x1,y1,x2,y2
                x, y, x2, y2 = parts
                w, h = x2-x, y2-y
            init_bbox = [x, y, w, h]
            
        # Track
        tracker = HiFTTracker(model)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if ret:
            tracker.init(frame, init_bbox)
            results.append({'sequence': seq_key, 'frame': 0, 
                            'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h})
            
            frame_id = 1
            while True:
                ret, frame = cap.read()
                if not ret: break
                try:
                    out = tracker.track(frame)
                    bbox = out['bbox'] # [x, y, w, h]
                    results.append({
                        'sequence': seq_key, 'frame': frame_id,
                        'x1': bbox[0], 'y1': bbox[1], 
                        'x2': bbox[0] + bbox[2], 'y2': bbox[1] + bbox[3]
                    })
                except Exception as e:
                    print(f"Error tracking frame {frame_id} in {seq_key}: {e}")
                    break
                frame_id += 1
        cap.release()
        
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv('submission.csv', index=False)
    print("✅ submission.csv generated!")

if __name__ == '__main__':
    run()