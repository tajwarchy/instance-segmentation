"""
Video file inference — runs Mask R-CNN on every frame of a video.
Usage:
    python inference/video_inference.py --input path/to/video.mp4
    python inference/video_inference.py --input path/to/video.mp4 --output results/visualizations/output.mp4
    python inference/video_inference.py --input path/to/video.mp4 --skip 2  # process every 2nd frame
"""

import sys
import time
import yaml
import torch
import cv2
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mask_rcnn import build_model, load_checkpoint
from inference.postprocess import postprocess_outputs
from inference.webcam_segmenter import preprocess_frame, render_frame
from training.visualization import CATEGORY_NAMES


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    cfg    = load_config(args.config)
    device = torch.device("cpu")
    size   = cfg["model"]["min_size"]

    print(f"\nDevice:  {device}")
    print(f"Input:   {args.input}")

    # ── Load model ──
    print("Loading model ...")
    model = build_model(
        num_classes = cfg["model"]["num_classes"],
        pretrained  = False,
        min_size    = size,
        max_size    = size,
    ).to(device)
    load_checkpoint(model, args.weights, device)
    model.eval()
    print("✅ Model loaded\n")

    # ── Open video ──
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"❌ Could not open video: {args.input}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS)
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video:   {src_w}x{src_h} @ {src_fps:.1f} fps — {total_frames} frames")

    # ── Output writer ──
    out_path = Path(args.output) if args.output else \
               Path(args.input).parent / f"{Path(args.input).stem}_segmented.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (src_w, src_h))

    score_thr = args.threshold if args.threshold else cfg["inference"]["score_threshold"]
    mask_thr  = cfg["inference"]["mask_threshold"]
    skip      = max(1, args.skip)

    # ── Stats ──
    frame_idx      = 0
    processed      = 0
    total_inst     = 0
    inference_times = []
    per_frame_stats = []

    print(f"Processing (every {skip} frame(s), threshold={score_thr:.2f}) ...\n")

    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc="  Frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip == 0:
                t0     = time.time()
                tensor = preprocess_frame(frame, size)
                out    = postprocess_outputs(
                    model([tensor.to(device)]),
                    score_thr   = score_thr,
                    mask_thr    = mask_thr,
                    nms_iou_thr = cfg["inference"]["nms_iou_threshold"],
                    img_size    = (size, size),
                )[0]
                elapsed = time.time() - t0
                inference_times.append(elapsed)

                canvas, instances = render_frame(frame, out, "mask", score_thr, mask_thr)

                # ── FPS overlay ──
                inf_fps = 1.0 / max(elapsed, 1e-6)
                cv2.rectangle(canvas, (0, 0), (src_w, 36), (20, 20, 20), -1)
                cv2.putText(canvas, f"FPS: {inf_fps:.1f}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
                cv2.putText(canvas, f"Instances: {len(instances)}", (130, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                cv2.putText(canvas, f"Frame: {frame_idx}/{total_frames}", (310, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

                total_inst += len(instances)
                processed  += 1

                per_frame_stats.append({
                    "frame":       frame_idx,
                    "n_instances": len(instances),
                    "inference_ms": round(elapsed * 1000, 1),
                    "instances": [
                        {"class": inst["class"], "score": inst["score"]}
                        for inst in instances
                    ],
                })

                last_canvas = canvas

            else:
                # Write last processed frame for skipped frames
                # (keeps output video smooth)
                canvas = last_canvas if 'last_canvas' in dir() else frame

            writer.write(canvas)
            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    writer.release()

    # ── Summary ──
    avg_ms  = sum(inference_times) / max(len(inference_times), 1) * 1000
    avg_fps = 1000.0 / max(avg_ms, 1e-6)

    print(f"\n{'='*50}")
    print(f"  Video inference complete")
    print(f"  Output:         {out_path}")
    print(f"  Frames:         {frame_idx} total, {processed} processed")
    print(f"  Avg inference:  {avg_ms:.1f} ms/frame ({avg_fps:.1f} FPS)")
    print(f"  Total instances detected: {total_inst}")
    print(f"  Avg instances/frame: {total_inst/max(processed,1):.1f}")
    print(f"{'='*50}")

    # ── Save per-frame report ──
    report = {
        "input":           args.input,
        "output":          str(out_path),
        "total_frames":    frame_idx,
        "processed_frames": processed,
        "avg_inference_ms": round(avg_ms, 1),
        "avg_fps":          round(avg_fps, 1),
        "total_instances":  total_inst,
        "per_frame":        per_frame_stats,
    }
    report_path = out_path.with_suffix(".json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {report_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video inference with Mask R-CNN")
    parser.add_argument("--input",     required=True,  help="Input video path")
    parser.add_argument("--output",    default=None,   help="Output video path")
    parser.add_argument("--weights",   default="weights/mask_rcnn_best.pth")
    parser.add_argument("--config",    default="configs/mask_rcnn_config.yaml")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Score threshold override")
    parser.add_argument("--skip",      type=int,   default=1,
                        help="Process every Nth frame (1=all, 2=every other, etc.)")
    args = parser.parse_args()
    main(args)