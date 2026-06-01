#!/usr/bin/env python3
"""Export all 8 sampled frames for 0034-necklace1 (qualitative panel c) for GT verification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STEM = "0034-necklace1"
FULL_ROOT = Path("/data/liuluyan/sam3/sam3_pvtt_full")
OUT_DIR = Path(__file__).resolve().parent / "necklace"
META_PATH = FULL_ROOT / "meta" / f"{STEM}.json"
GT_THRESHOLD = 127
OVERLAY_ALPHA = 0.4


def load_gt_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > GT_THRESHOLD).astype(np.uint8)


def gt_only_overlay(rgb: np.ndarray, gt: np.ndarray) -> np.ndarray:
    base = rgb.astype(np.float32)
    out = base.copy()
    gt_c = np.zeros_like(base)
    gt_c[:, :, 1] = 255.0
    gt_m = gt.astype(bool)
    out[gt_m] = (1 - OVERLAY_ALPHA) * base[gt_m] + OVERLAY_ALPHA * gt_c[gt_m]
    return np.clip(out, 0, 255).astype(np.uint8)


def pred_only_overlay(rgb: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = rgb.astype(np.float32)
    out = base.copy()
    pr_c = np.zeros_like(base)
    pr_c[:, :, 2] = 255.0
    pr_m = pred.astype(bool)
    out[pr_m] = (1 - OVERLAY_ALPHA) * base[pr_m] + OVERLAY_ALPHA * pr_c[pr_m]
    return np.clip(out, 0, 255).astype(np.uint8)


def hstack_images(images: list[Image.Image], labels: list[str], gap: int = 8) -> Image.Image:
    font = ImageFont.load_default()
    label_h = 18
    max_h = max(im.height for im in images)
    total_w = sum(im.width for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (total_w, max_h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, label in zip(images, labels):
        y_off = (max_h - im.height) // 2
        canvas.paste(im, (x, label_h + y_off))
        draw.text((x + 4, 2), label, fill=(0, 0, 0), font=font)
        x += im.width + gap
    return canvas


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for fr in meta["frames"]:
        k = int(fr["frame_idx"])
        tag = f"{k:02d}"
        frame_path = FULL_ROOT / "frames" / STEM / f"frame_{tag}.jpg"
        gt_path = Path(fr["gt_path"])
        pred_path = FULL_ROOT / "pred_masks" / STEM / f"frame_{tag}.png"
        overlay_path = FULL_ROOT / "overlays" / STEM / f"frame_{tag}.jpg"

        rgb = np.array(Image.open(frame_path).convert("RGB"))
        gt_bin = load_gt_binary(gt_path)
        h, w = gt_bin.shape
        rgb_resized = np.array(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR))

        gt_only = gt_only_overlay(rgb_resized, gt_bin)
        pred_bin = (np.array(Image.open(pred_path).convert("L")) > 127).astype(np.uint8)
        pred_only = pred_only_overlay(rgb_resized, pred_bin)

        # Individual exports
        shutil.copy2(frame_path, OUT_DIR / f"frame_{tag}.jpg")
        shutil.copy2(gt_path, OUT_DIR / f"gt_mask_{tag}.jpg")
        shutil.copy2(pred_path, OUT_DIR / f"pred_mask_{tag}.png")
        shutil.copy2(overlay_path, OUT_DIR / f"overlay_gt_pred_{tag}.jpg")
        Image.fromarray(gt_only).save(OUT_DIR / f"overlay_gt_only_{tag}.jpg", quality=92)
        Image.fromarray(pred_only).save(OUT_DIR / f"overlay_pred_only_{tag}.jpg", quality=92)

        compare = hstack_images(
            [
                Image.fromarray(rgb_resized),
                Image.fromarray(gt_only),
                Image.fromarray(pred_only),
                Image.open(overlay_path).convert("RGB"),
            ],
            ["frame", "GT (green)", "pred (red)", "GT+pred"],
        )
        compare.save(OUT_DIR / f"compare_{tag}.jpg", quality=92)

        summary_rows.append(
            {
                "frame_idx": k,
                "time_sec": fr["time"],
                "gt_frame_idx": fr["gt_frame"],
                "gt_path": fr["gt_path"],
                "iou": fr["iou"],
                "files": {
                    "frame": f"frame_{tag}.jpg",
                    "gt_mask": f"gt_mask_{tag}.jpg",
                    "pred_mask": f"pred_mask_{tag}.png",
                    "overlay_gt_only": f"overlay_gt_only_{tag}.jpg",
                    "overlay_pred_only": f"overlay_pred_only_{tag}.jpg",
                    "overlay_gt_pred": f"overlay_gt_pred_{tag}.jpg",
                    "compare": f"compare_{tag}.jpg",
                },
            }
        )

    summary = {
        "video": STEM,
        "category": meta["category"],
        "prompt": meta["prompt"],
        "video_path": meta["video_path"],
        "n_gt_frames_in_dataset": meta["n_gt_frames"],
        "mean_iou": meta["mean_iou"],
        "median_iou": meta["median_iou"],
        "note": "Green=GT, Red=SAM3 prediction. Panel (c) in sam3_qualitative.pdf uses frame_01.",
        "frames": summary_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(summary_rows)} frames to {OUT_DIR}")


if __name__ == "__main__":
    main()
