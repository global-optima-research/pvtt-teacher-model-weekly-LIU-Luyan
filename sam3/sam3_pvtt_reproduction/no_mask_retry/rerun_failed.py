#!/usr/bin/env python3
"""Retry INFER_NO_MASK failures with alternate prompts/thresholds; copy all other results."""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WORK_ROOT = Path(__file__).resolve().parent
ORIG_ROOT = WORK_ROOT.parent
SAM3_REPO = Path("/data/liuluyan/sam3")
DEMO_SCRIPT_DIR = SAM3_REPO / "sam3_pvtt_demo"
DATA_ROOT = Path("/data/datasets/pvtt_new/pvtt-evaluation/datasets_new")
MASK_ROOT = DATA_ROOT / "masks"

GT_THRESHOLD = 127
N_FRAMES = 8
OVERLAY_ALPHA = 0.4

FAILED_BY_STEM: dict[str, list[int]] = {
    "0013-handbag3_scene01": [1, 2, 3, 5, 6, 7],
    "0013-handbag3_scene02": list(range(8)),
    "0013-handbag3_scene03": list(range(8)),
    "0038-necklace5": [4],
}

# README suggested retry strategies (scene01 uses same strategy as scene02)
RETRY_ATTEMPTS: dict[str, list[dict]] = {
    "0013-handbag3_scene03": [
        {"prompt": "backpack", "threshold": 0.1, "label": "backpack@0.1"},
        {"prompt": "backpack", "threshold": 0.05, "label": "backpack@0.05"},
    ],
    "0013-handbag3_scene02": [
        {"prompt": "messenger bag", "threshold": 0.1, "label": "messenger bag@0.1"},
        {"prompt": "sling bag", "threshold": 0.1, "label": "sling bag@0.1"},
        {"prompt": "handbag", "threshold": 0.05, "label": "handbag@0.05"},
        {"prompt": "handbag", "threshold": 0.01, "label": "handbag@0.01"},
    ],
    "0013-handbag3_scene01": [
        {"prompt": "messenger bag", "threshold": 0.1, "label": "messenger bag@0.1"},
        {"prompt": "sling bag", "threshold": 0.1, "label": "sling bag@0.1"},
        {"prompt": "handbag", "threshold": 0.05, "label": "handbag@0.05"},
        {"prompt": "handbag", "threshold": 0.01, "label": "handbag@0.01"},
    ],
    "0038-necklace5": [
        {"prompt": "necklace", "threshold": 0.1, "label": "necklace@0.1"},
        {"prompt": "necklace", "threshold": 0.05, "label": "necklace@0.05"},
        {"prompt": "necklace", "threshold": 0.01, "label": "necklace@0.01"},
    ],
}

CATEGORY_RE = re.compile(r"\d+-([a-z]+)")
SCENE_RE = re.compile(r"_scene(\d+)")

sys.path.insert(0, str(SAM3_REPO))
sys.path.insert(0, str(DEMO_SCRIPT_DIR))
import test_sam3  # noqa: E402

test_sam3.WEIGHTS_DIR = SAM3_REPO / "weights"
test_sam3.CHECKPOINT_CANDIDATES = [
    test_sam3.WEIGHTS_DIR / "sam3.pt",
    test_sam3.WEIGHTS_DIR / "model.safetensors",
]


def parse_category(stem: str) -> str:
    m = CATEGORY_RE.search(stem)
    return m.group(1) if m else "unknown"


def parse_scene(stem: str) -> str | None:
    m = SCENE_RE.search(stem)
    return m.group(1) if m else None


def gt_path_for(stem: str, frame_idx: int) -> Path:
    frames = sorted((MASK_ROOT / stem).glob("*.jpg"))
    n_gt = len(frames)
    ratio = (frame_idx + 0.5) / N_FRAMES
    gt_idx = int(round(ratio * max(n_gt - 1, 0)))
    return frames[max(0, min(gt_idx, n_gt - 1))]


def frame_time(stem: str, frame_idx: int) -> float:
    meta_path = ORIG_ROOT / "meta" / f"{stem}.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for fr in meta.get("frames", []):
            if int(fr["frame_idx"]) == frame_idx:
                return float(fr["time"])
    import subprocess

    video = DATA_ROOT / "source_videos_100" / f"{stem}.mp4"
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ]
    dur = float(subprocess.check_output(cmd, text=True).strip())
    return dur * (frame_idx + 0.5) / N_FRAMES


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


def copy_baseline() -> None:
    for sub in ("pred_masks", "overlays", "meta"):
        (WORK_ROOT / sub).mkdir(parents=True, exist_ok=True)

    frames_link = WORK_ROOT / "frames"
    if not frames_link.exists():
        frames_link.symlink_to(ORIG_ROOT / "frames", target_is_directory=True)

    partial = set(FAILED_BY_STEM)
    for meta_file in sorted((ORIG_ROOT / "meta").glob("*.json")):
        stem = meta_file.stem
        if stem not in partial:
            shutil.copy2(meta_file, WORK_ROOT / "meta" / meta_file.name)
            for sub in ("pred_masks", "overlays"):
                src_dir = ORIG_ROOT / sub / stem
                if src_dir.is_dir():
                    dst_dir = WORK_ROOT / sub / stem
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
            continue

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        ok_indices = {int(fr["frame_idx"]) for fr in meta.get("frames", [])}
        for k in ok_indices:
            for sub, ext in (("pred_masks", ".png"), ("overlays", ".jpg")):
                src = ORIG_ROOT / sub / stem / f"frame_{k:02d}{ext}"
                dst = WORK_ROOT / sub / stem / f"frame_{k:02d}{ext}"
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)


def load_model():
    device = test_sam3.pick_gpu()
    if device == "cuda" and torch.cuda.device_count() > 1:
        # Prefer a GPU with enough free memory when default GPU is occupied.
        best_idx = 0
        best_free = -1
        for idx in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(idx)
            if free > best_free:
                best_free = free
                best_idx = idx
        if best_idx != torch.cuda.current_device():
            torch.cuda.set_device(best_idx)
            device = f"cuda:{best_idx}"
    ckpt = test_sam3.resolve_checkpoint()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"Device: {device}, checkpoint: {ckpt}", flush=True)
    model = build_sam3_image_model(device=device, checkpoint_path=ckpt, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=0.1)
    return processor, device


def infer_frame(processor, frame_path: Path, prompt: str, threshold: float):
    processor.set_confidence_threshold(threshold)
    image = Image.open(frame_path).convert("RGB")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)
    pred_bin = masks_to_binary(output.get("masks"))
    n_masks = len(output.get("masks", [])) if output.get("masks") is not None else 0
    return pred_bin, n_masks, image


def save_frame_outputs(stem: str, k: int, image: Image.Image, gt_bin: np.ndarray, pred_bin: np.ndarray, meta_extra: dict) -> dict:
    h, w = gt_bin.shape
    pred_resized = resize_mask_nearest(pred_bin, (w, h))
    iou = compute_iou(pred_resized, gt_bin)

    pred_out = WORK_ROOT / "pred_masks" / stem / f"frame_{k:02d}.png"
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((pred_resized * 255).astype(np.uint8)).save(pred_out)

    rgb = np.array(image.resize((w, h), Image.Resampling.BILINEAR))
    overlay = make_overlay(rgb, gt_bin, pred_resized)
    overlay_path = WORK_ROOT / "overlays" / stem / f"frame_{k:02d}.jpg"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(overlay_path, quality=92)

    return {
        "frame_idx": k,
        "time": round(frame_time(stem, k), 4),
        "gt_frame": int(meta_extra.get("gt_frame", 0)),
        "gt_path": meta_extra["gt_path"],
        "prompt": meta_extra["prompt"],
        "confidence_threshold": meta_extra["threshold"],
        "retry_label": meta_extra["label"],
        "iou": round(iou, 4),
        "n_masks": meta_extra["n_masks"],
        "overlay": str(overlay_path),
        "attempt_log": meta_extra["attempt_log"],
    }


def retry_failed(processor) -> dict:
    results: dict = {"frames": [], "summary": {}}
    recovered = 0
    still_failed = 0

    for stem, fail_indices in FAILED_BY_STEM.items():
        attempts_cfg = RETRY_ATTEMPTS[stem]
        for k in fail_indices:
            frame_path = ORIG_ROOT / "frames" / stem / f"frame_{k:02d}.jpg"
            gt_path = gt_path_for(stem, k)
            gt_bin = load_gt_binary(gt_path)
            gt_idx = sorted((MASK_ROOT / stem).glob("*.jpg")).index(gt_path)

            attempt_log = []
            success = None
            for att in attempts_cfg:
                t0 = time.time()
                pred_bin, n_masks, image = infer_frame(
                    processor, frame_path, att["prompt"], att["threshold"]
                )
                sec = round(time.time() - t0, 3)
                got_mask = pred_bin is not None
                attempt_log.append(
                    {
                        "label": att["label"],
                        "prompt": att["prompt"],
                        "threshold": att["threshold"],
                        "got_mask": got_mask,
                        "n_masks": n_masks,
                        "infer_sec": sec,
                    }
                )
                if got_mask and success is None:
                    success = {
                        **att,
                        "attempt_log": list(attempt_log),
                        "gt_path": str(gt_path),
                        "gt_frame": gt_idx,
                    }

            entry = {
                "video": stem,
                "frame_idx": k,
                "original_failure": "INFER_NO_MASK",
                "attempt_log": attempt_log,
            }
            if success:
                pred_bin, n_masks, image = infer_frame(
                    processor, frame_path, success["prompt"], success["threshold"]
                )
                success["n_masks"] = n_masks
                fr_meta = save_frame_outputs(stem, k, image, gt_bin, pred_bin, success)
                entry["recovered"] = True
                entry["winning_attempt"] = success["label"]
                entry["iou"] = fr_meta["iou"]
                entry["prompt"] = success["prompt"]
                entry["threshold"] = success["threshold"]
                recovered += 1
            else:
                entry["recovered"] = False
                still_failed += 1
            results["frames"].append(entry)
            print(
                f"  {stem} frame_{k:02d}: "
                f"{'OK ' + entry.get('winning_attempt', '') if entry['recovered'] else 'FAIL'}",
                flush=True,
            )

    results["summary"] = {
        "total_retried": sum(len(v) for v in FAILED_BY_STEM.values()),
        "recovered": recovered,
        "still_failed": still_failed,
        "recovery_rate": round(recovered / max(sum(len(v) for v in FAILED_BY_STEM.values()), 1), 4),
    }
    return results


def rebuild_partial_meta(retry_results: dict) -> None:
    for stem in FAILED_BY_STEM:
        orig_meta_path = ORIG_ROOT / "meta" / f"{stem}.json"
        orig_meta = json.loads(orig_meta_path.read_text(encoding="utf-8"))
        frames = list(orig_meta.get("frames", []))
        existing = {int(fr["frame_idx"]) for fr in frames}

        for entry in retry_results["frames"]:
            if entry["video"] != stem or not entry["recovered"]:
                continue
            k = int(entry["frame_idx"])
            if k in existing:
                continue
            gt_path = gt_path_for(stem, k)
            frames.append(
                {
                    "frame_idx": k,
                    "time": round(frame_time(stem, k), 4),
                    "gt_frame": sorted((MASK_ROOT / stem).glob("*.jpg")).index(gt_path),
                    "gt_path": str(gt_path),
                    "prompt": entry["prompt"],
                    "iou": entry["iou"],
                    "n_masks": next(a["n_masks"] for a in entry["attempt_log"] if a["got_mask"]),
                    "infer_sec": next(a["infer_sec"] for a in entry["attempt_log"] if a["got_mask"]),
                    "overlay": str(WORK_ROOT / "overlays" / stem / f"frame_{k:02d}.jpg"),
                    "retry_label": entry["winning_attempt"],
                    "confidence_threshold": entry["threshold"],
                }
            )

        frames.sort(key=lambda x: int(x["frame_idx"]))
        ious = [float(fr["iou"]) for fr in frames]
        meta = {
            **{k: v for k, v in orig_meta.items() if k not in ("frames", "mean_iou", "median_iou", "n_success", "n_failed")},
            "frames": frames,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "median_iou": float(np.median(ious)) if ious else 0.0,
            "n_success": len(frames),
            "n_failed": N_FRAMES - len(frames),
            "retry_note": "Partial re-run for previously failed frames; other frames copied from sam3_pvtt_full.",
        }
        (WORK_ROOT / "meta" / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    print("Step 1: copy baseline results ...", flush=True)
    copy_baseline()

    print("Step 2: load model ...", flush=True)
    processor, _ = load_model()

    print("Step 3: retry failed frames ...", flush=True)
    retry_results = retry_failed(processor)
    (WORK_ROOT / "retry_results.json").write_text(
        json.dumps(retry_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Step 4: rebuild meta for partial videos ...", flush=True)
    rebuild_partial_meta(retry_results)

    s = retry_results["summary"]
    print(
        f"\nDone. recovered={s['recovered']}/{s['total_retried']}  "
        f"still_failed={s['still_failed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
