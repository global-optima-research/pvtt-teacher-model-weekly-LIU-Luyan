#!/usr/bin/env python3
"""SAM3 full PVTT eval: source_videos_100, 8 frames/video, resume via progress.json."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

WORK_ROOT = Path(__file__).resolve().parent
SAM3_REPO = Path("/data/liuluyan/sam3")
DEMO_SCRIPT_DIR = SAM3_REPO / "sam3_pvtt_demo"
DATA_ROOT = Path("/data/datasets/pvtt_new/pvtt-evaluation/datasets_new")
VIDEO_ROOT = DATA_ROOT / "source_videos_100"
MASK_ROOT = DATA_ROOT / "masks"

FRAMES_DIR = WORK_ROOT / "frames"
PRED_DIR = WORK_ROOT / "pred_masks"
OVERLAY_DIR = WORK_ROOT / "overlays"
META_DIR = WORK_ROOT / "meta"
MANIFEST_PATH = WORK_ROOT / "manifest.csv"
IOU_SUMMARY_PATH = WORK_ROOT / "iou_summary.csv"
FAILED_PATH = WORK_ROOT / "failed.txt"
PROGRESS_PATH = WORK_ROOT / "progress.json"

GT_THRESHOLD = 127
N_FRAMES = 8
OVERLAY_ALPHA = 0.4

CATEGORY_RE = re.compile(r"\d+-([a-z]+)")
SCENE_RE = re.compile(r"_scene(\d+)")

# Prompt map: parsed category -> SAM3 text prompt (watch uses wristwatch per minimal demo)
CATEGORY_PROMPT_MAP: dict[str, str] = {
    "bracelet": "bracelet",
    "earring": "earring",
    "handbag": "handbag",
    "handfan": "handfan",
    "necklace": "necklace",
    "purse": "purse",
    "sunglasses": "sunglasses",
    "watch": "wristwatch",
    "clothing": "clothing",
}

sys.path.insert(0, str(SAM3_REPO))
sys.path.insert(0, str(DEMO_SCRIPT_DIR))
import test_sam3  # noqa: E402

# Weights live under SAM3 repo, not sam3_pvtt_demo/
test_sam3.WEIGHTS_DIR = SAM3_REPO / "weights"
test_sam3.CHECKPOINT_CANDIDATES = [
    test_sam3.WEIGHTS_DIR / "sam3.pt",
    test_sam3.WEIGHTS_DIR / "model.safetensors",
]


def log_fail(msg: str) -> None:
    with FAILED_PATH.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def load_progress() -> dict:
    if PROGRESS_PATH.is_file():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {"completed": [], "version": 1}


def save_progress(progress: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def parse_category(stem: str) -> str:
    m = CATEGORY_RE.search(stem)
    return m.group(1) if m else "unknown"


def parse_scene(stem: str) -> str | None:
    m = SCENE_RE.search(stem)
    return m.group(1) if m else None


def prompt_for_stem(stem: str) -> str:
    cat = parse_category(stem)
    return CATEGORY_PROMPT_MAP.get(cat, cat)


def list_eval_videos() -> list[str]:
    stems = []
    for mp4 in sorted(VIDEO_ROOT.glob("*.mp4")):
        stem = mp4.stem
        if (MASK_ROOT / stem).is_dir():
            stems.append(stem)
        else:
            log_fail(f"SKIP_NO_MASK_DIR {stem}")
    return stems


def video_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    return float(subprocess.check_output(cmd, text=True).strip())


def extract_frame(video: Path, t: float, out_jpg: Path) -> None:
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{t:.4f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_jpg),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def gt_frame_path(stem: str, gt_idx: int) -> Path:
    mask_dir = MASK_ROOT / stem
    frames = sorted(mask_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No GT frames in {mask_dir}")
    idx = max(0, min(gt_idx, len(frames) - 1))
    return frames[idx]


def load_gt_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > GT_THRESHOLD).astype(np.uint8)


def resize_mask_nearest(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    resized = img.resize(size_wh, Image.Resampling.NEAREST)
    return (np.array(resized) > 127).astype(np.uint8)


def masks_to_binary(pred_masks) -> np.ndarray | None:
    if pred_masks is None or len(pred_masks) == 0:
        return None
    combined = None
    for m in pred_masks:
        if isinstance(m, torch.Tensor):
            arr = m.detach().float().cpu().numpy()
        else:
            arr = np.asarray(m)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        bin_m = (arr > 0).astype(np.uint8)
        combined = bin_m if combined is None else np.maximum(combined, bin_m)
    return combined


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def make_overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = rgb.astype(np.float32)
    out = base.copy()
    gt_c = np.zeros_like(base)
    gt_c[:, :, 1] = 255.0
    pr_c = np.zeros_like(base)
    pr_c[:, :, 2] = 255.0
    gt_m = gt.astype(bool)
    pr_m = pred.astype(bool)
    out[gt_m] = (1 - OVERLAY_ALPHA) * base[gt_m] + OVERLAY_ALPHA * gt_c[gt_m]
    out[pr_m] = (1 - OVERLAY_ALPHA) * out[pr_m] + OVERLAY_ALPHA * pr_c[pr_m]
    return np.clip(out, 0, 255).astype(np.uint8)


def run_sam3_on_frame(processor, image: Image.Image, prompt: str):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)
    return output


def load_model_and_processor():
    device = test_sam3.pick_gpu()
    ckpt = test_sam3.resolve_checkpoint()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"Device: {device}, checkpoint: {ckpt}", flush=True)
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=ckpt,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=0.1)
    return processor, device


def prepare_video_rows(stem: str, video_path: Path) -> list[dict]:
    dur = video_duration(video_path)
    n_gt = len(list((MASK_ROOT / stem).glob("*.jpg")))
    cat = parse_category(stem)
    scene = parse_scene(stem)
    prompt = prompt_for_stem(stem)
    rows = []
    for k in range(N_FRAMES):
        ratio = (k + 0.5) / N_FRAMES
        t = dur * ratio
        gt_idx = int(round(ratio * max(n_gt - 1, 0)))
        gt_path = gt_frame_path(stem, gt_idx)
        frame_path = FRAMES_DIR / stem / f"frame_{k:02d}.jpg"
        try:
            if not frame_path.is_file():
                extract_frame(video_path, t, frame_path)
        except Exception as e:
            log_fail(f"EXTRACT_FAIL {frame_path}: {e}")
            continue
        rows.append(
            {
                "video": stem,
                "category": cat,
                "scene": scene or "",
                "frame_idx": k,
                "time": round(t, 4),
                "gt_frame": gt_idx,
                "gt_path": str(gt_path),
                "prompt": prompt,
                "frame_path": str(frame_path),
            }
        )
    return rows


def process_one_video(processor, device: str, stem: str) -> dict:
    video_path = VIDEO_ROOT / f"{stem}.mp4"
    cat = parse_category(stem)
    scene = parse_scene(stem)
    prompt = prompt_for_stem(stem)
    rows = prepare_video_rows(stem, video_path)

    frame_metas = []
    n_failed = 0
    for row in rows:
        k = int(row["frame_idx"])
        frame_path = Path(row["frame_path"])
        if not frame_path.is_file():
            n_failed += 1
            continue
        t0 = time.time()
        try:
            image = Image.open(frame_path).convert("RGB")
            output = run_sam3_on_frame(processor, image, prompt)
            infer_sec = time.time() - t0
            pred_bin = masks_to_binary(output.get("masks"))
            if pred_bin is None:
                log_fail(f"INFER_NO_MASK {frame_path}")
                n_failed += 1
                continue

            gt_path = Path(row["gt_path"])
            gt_bin = load_gt_binary(gt_path)
            h, w = gt_bin.shape
            pred_resized = resize_mask_nearest(pred_bin, (w, h))
            iou = compute_iou(pred_resized, gt_bin)

            pred_out = PRED_DIR / stem / f"frame_{k:02d}.png"
            pred_out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((pred_resized * 255).astype(np.uint8)).save(pred_out)

            rgb = np.array(image.resize((w, h), Image.Resampling.BILINEAR))
            overlay = make_overlay(rgb, gt_bin, pred_resized)
            overlay_path = OVERLAY_DIR / stem / f"frame_{k:02d}.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay).save(overlay_path, quality=92)

            frame_metas.append(
                {
                    "frame_idx": k,
                    "time": row["time"],
                    "gt_frame": row["gt_frame"],
                    "gt_path": row["gt_path"],
                    "prompt": prompt,
                    "iou": round(iou, 4),
                    "n_masks": len(output.get("masks", [])),
                    "infer_sec": round(infer_sec, 3),
                    "overlay": str(overlay_path),
                }
            )
        except Exception as e:
            log_fail(f"FRAME_FAIL {frame_path}: {e}\n{traceback.format_exc()}")
            n_failed += 1

    ious = [f["iou"] for f in frame_metas]
    mean_iou = float(np.mean(ious)) if ious else 0.0
    median_iou = float(np.median(ious)) if ious else 0.0
    meta = {
        "video": stem,
        "category": cat,
        "scene": scene,
        "prompt": prompt,
        "video_path": str(video_path),
        "n_gt_frames": len(list((MASK_ROOT / stem).glob("*.jpg"))),
        "frames": frame_metas,
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "n_success": len(frame_metas),
        "n_failed": n_failed,
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def rebuild_manifest_and_summary(all_stems: list[str]) -> None:
    manifest_rows = []
    summary_rows = []
    for stem in all_stems:
        meta_path = META_DIR / f"{stem}.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cat = meta.get("category", parse_category(stem))
        scene = meta.get("scene") or ""
        prompt = meta.get("prompt", prompt_for_stem(stem))
        for fr in meta.get("frames", []):
            manifest_rows.append(
                {
                    "video": stem,
                    "category": cat,
                    "scene": scene,
                    "frame_idx": fr["frame_idx"],
                    "time": fr["time"],
                    "gt_frame": fr["gt_frame"],
                    "gt_path": fr["gt_path"],
                    "prompt": prompt,
                    "frame_path": str(FRAMES_DIR / stem / f"frame_{int(fr['frame_idx']):02d}.jpg"),
                }
            )
        summary_rows.append(
            {
                "video": stem,
                "category": cat,
                "scene": scene,
                "prompt": prompt,
                "mean_iou": round(float(meta.get("mean_iou", 0)), 4),
                "median_iou": round(float(meta.get("median_iou", 0)), 4),
                "n_frames": int(meta.get("n_success", 0)),
                "n_failed": int(meta.get("n_failed", 0)),
            }
        )

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "category",
                "scene",
                "frame_idx",
                "time",
                "gt_frame",
                "gt_path",
                "prompt",
                "frame_path",
            ],
        )
        w.writeheader()
        w.writerows(manifest_rows)

    with IOU_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "category",
                "scene",
                "prompt",
                "mean_iou",
                "median_iou",
                "n_frames",
                "n_failed",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)


def main() -> int:
    t0 = time.time()
    if not FAILED_PATH.exists():
        FAILED_PATH.write_text("", encoding="utf-8")

    all_stems = list_eval_videos()
    print(f"Eval videos (Plan A): {len(all_stems)} x {N_FRAMES} frames = up to {len(all_stems)*N_FRAMES}", flush=True)

    progress = load_progress()
    completed = set(progress.get("completed", []))
    todo = [s for s in all_stems if s not in completed]
    print(f"Resume: {len(completed)} done, {len(todo)} remaining", flush=True)

    processor, device = load_model_and_processor()

    for i, stem in enumerate(tqdm(todo, desc="videos"), start=len(completed) + 1):
        try:
            meta = process_one_video(processor, device, stem)
            completed.add(stem)
            progress["completed"] = sorted(completed)
            save_progress(progress)
            print(
                f"[{i}/{len(all_stems)}] {stem}  cat={meta['category']}  "
                f"prompt={meta['prompt']}  n_ok={meta['n_success']}  "
                f"mean_iou={meta['mean_iou']:.3f}  failed={meta['n_failed']}",
                flush=True,
            )
        except Exception as e:
            log_fail(f"VIDEO_FAIL {stem}: {e}\n{traceback.format_exc()}")
            print(f"[{i}/{len(all_stems)}] {stem} VIDEO_FAIL: {e}", flush=True)

    rebuild_manifest_and_summary(all_stems)
    elapsed = time.time() - t0
    print(f"\nDone. elapsed={elapsed:.1f}s", flush=True)
    print(f"manifest: {MANIFEST_PATH}", flush=True)
    print(f"iou_summary: {IOU_SUMMARY_PATH}", flush=True)
    print(f"progress: {PROGRESS_PATH} ({len(completed)}/{len(all_stems)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
