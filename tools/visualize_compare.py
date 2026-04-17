import cv2
import torch
import numpy as np


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.hift_tracker import HiFTTracker


# ---------------- LOAD MODEL ----------------
def load_model(path):
    model = ModelBuilder()
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.cuda().eval()
    return model


# ---------------- DRAW BOX ----------------
def draw_bbox(frame, bbox, color, label):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


# ---------------- TRACK VIDEO ----------------
def track_video(model, video_path, init_bbox, n_frames):
    tracker = HiFTTracker(model)

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()

    tracker.init(frame, init_bbox)

    results = [init_bbox]

    for _ in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        out = tracker.track(frame)
        bbox = out['bbox']
        results.append(bbox)

    cap.release()
    return results


# ---------------- MAIN VISUALIZATION ----------------
def compare(pretrained_path, finetuned_path,
            video_path, init_bbox, output_path):

    cfg.merge_from_file("configs/hiFT_finetune.yaml")

    pretrained = load_model(pretrained_path)
    finetuned  = load_model(finetuned_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # trackers
    t_pre = HiFTTracker(pretrained)
    t_ft  = HiFTTracker(finetuned)

    ret, frame = cap.read()
    if not ret:
        return

    t_pre.init(frame, init_bbox)
    t_ft.init(frame, init_bbox)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pre_bbox = t_pre.track(frame)['bbox']
        ft_bbox  = t_ft.track(frame)['bbox']

        vis = frame.copy()

        # pretrained = RED
        vis = draw_bbox(vis, pre_bbox, (0, 0, 255), "Pretrained")

        # fine-tuned = GREEN
        vis = draw_bbox(vis, ft_bbox, (0, 255, 0), "Fine-tuned")

        out.write(vis)

        frame_idx += 1

    cap.release()
    out.release()

    print(f"Saved comparison video → {output_path}")


# ---------------- EXAMPLE RUN ----------------
if __name__ == "__main__":
    compare(
        pretrained_path="pretrained/first.pth",
        finetuned_path="snapshot/best.pth",
        video_path = r"..\AIC4-UAV-Tracker\data\contest_release\dataset1\basketball\basketball.mp4",
        init_bbox=[625,381,73,57],
        output_path="comparison.mp4"
    )