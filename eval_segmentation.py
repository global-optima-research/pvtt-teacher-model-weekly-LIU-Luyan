"""
DAVIS 2017 Segmentation Evaluation Script
==========================================
Evaluates 5 segmentation models against DAVIS 2017 val ground truth.

Metrics:
  1. J (Jaccard / IoU) - Region similarity
  2. F (Boundary F-measure) - Contour accuracy
  3. J&F - Overall score (standard DAVIS metric)
  4. Temporal Consistency - Adjacent frame IoU
  5. Area Stability - Frame-to-frame mask area change ratio

Usage:
  python eval_segmentation.py --davis_root /data/datasets/DAVIS

Output:
  - Per-video CSV results
  - Per-model summary table
  - JSON with all detailed metrics
"""

import os
import argparse
import json
import csv
import numpy as np
from PIL import Image
from collections import defaultdict


# ============================================================
# Metric 1: J (Jaccard Index / IoU)
# ============================================================
def compute_jaccard(pred, gt):
    """Compute IoU between predicted and ground truth masks."""
    # Handle multi-object: compute per-object then average
    obj_ids = np.unique(gt)
    obj_ids = obj_ids[obj_ids > 0]  # exclude background

    if len(obj_ids) == 0:
        return 1.0 if np.sum(pred > 0) == 0 else 0.0

    j_scores = []
    for oid in obj_ids:
        pred_mask = (pred == oid)
        gt_mask = (gt == oid)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            j_scores.append(1.0)
        else:
            j_scores.append(intersection / union)
    return np.mean(j_scores)


# ============================================================
# Metric 2: F (Boundary F-measure)
# ============================================================
def mask_to_boundary(mask, dilation=5):
    """Convert a binary mask to its boundary pixels."""
    from scipy.ndimage import binary_dilation, binary_erosion
    eroded = binary_erosion(mask, iterations=1)
    boundary = mask.astype(bool) ^ eroded.astype(bool)
    return boundary


def compute_boundary_f(pred, gt, dilation=5):
    """Compute boundary F-measure between predicted and GT masks."""
    from scipy.ndimage import binary_dilation

    obj_ids = np.unique(gt)
    obj_ids = obj_ids[obj_ids > 0]

    if len(obj_ids) == 0:
        return 1.0 if np.sum(pred > 0) == 0 else 0.0

    f_scores = []
    for oid in obj_ids:
        pred_boundary = mask_to_boundary(pred == oid)
        gt_boundary = mask_to_boundary(gt == oid)

        # Dilate boundaries for tolerance matching
        pred_dilated = binary_dilation(pred_boundary, iterations=dilation)
        gt_dilated = binary_dilation(gt_boundary, iterations=dilation)

        # Precision: how many predicted boundary pixels are near GT boundary
        if pred_boundary.sum() == 0:
            precision = 1.0 if gt_boundary.sum() == 0 else 0.0
        else:
            precision = np.logical_and(pred_boundary, gt_dilated).sum() / pred_boundary.sum()

        # Recall: how many GT boundary pixels are near predicted boundary
        if gt_boundary.sum() == 0:
            recall = 1.0 if pred_boundary.sum() == 0 else 0.0
        else:
            recall = np.logical_and(gt_boundary, pred_dilated).sum() / gt_boundary.sum()

        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            f_scores.append(2 * precision * recall / (precision + recall))

    return np.mean(f_scores)


# ============================================================
# Metric 3: Temporal Consistency (adjacent frame IoU)
# ============================================================
def compute_temporal_consistency(masks):
    """Compute mean IoU between adjacent frames.
    masks: list of numpy arrays (frame masks in order).
    """
    if len(masks) < 2:
        return 1.0

    ious = []
    for i in range(len(masks) - 1):
        m1 = masks[i] > 0
        m2 = masks[i + 1] > 0
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return np.mean(ious)


# ============================================================
# Metric 4: Area Stability (frame-to-frame area change)
# ============================================================
def compute_area_stability(masks):
    """Compute max frame-to-frame area change ratio.
    Returns (mean_change, max_change, num_frames_exceeding_20pct).
    """
    if len(masks) < 2:
        return 0.0, 0.0, 0

    areas = [np.sum(m > 0) for m in masks]
    changes = []
    exceed_count = 0
    for i in range(len(areas) - 1):
        if areas[i] == 0:
            change = 0.0 if areas[i + 1] == 0 else 1.0
        else:
            change = abs(areas[i + 1] - areas[i]) / areas[i]
        changes.append(change)
        if change > 0.2:
            exceed_count += 1

    return np.mean(changes), np.max(changes), exceed_count


# ============================================================
# Main evaluation
# ============================================================
def load_mask(path):
    """Load a mask image as numpy array."""
    img = Image.open(path)
    return np.array(img)


def evaluate_model(model_name, model_dir, gt_dir, jpeg_dir, videos):
    """Evaluate a single model across all videos."""
    results = []

    for video in videos:
        gt_video_dir = os.path.join(gt_dir, video)
        pred_video_dir = os.path.join(model_dir, video)

        if not os.path.isdir(pred_video_dir):
            print(f"  [SKIP] {model_name}/{video}: prediction dir not found")
            continue

        gt_frames = sorted([f for f in os.listdir(gt_video_dir) if f.endswith('.png')])
        pred_frames = sorted([f for f in os.listdir(pred_video_dir) if f.endswith('.png')])

        j_scores = []
        f_scores = []
        pred_masks_all = []

        for frame_name in gt_frames:
            gt_path = os.path.join(gt_video_dir, frame_name)
            pred_path = os.path.join(pred_video_dir, frame_name)

            gt_mask = load_mask(gt_path)

            if os.path.exists(pred_path):
                pred_mask = load_mask(pred_path)
            else:
                pred_mask = np.zeros_like(gt_mask)

            j = compute_jaccard(pred_mask, gt_mask)
            f = compute_boundary_f(pred_mask, gt_mask)
            j_scores.append(j)
            f_scores.append(f)
            pred_masks_all.append(pred_mask)

        mean_j = np.mean(j_scores) if j_scores else 0.0
        mean_f = np.mean(f_scores) if f_scores else 0.0
        jf = (mean_j + mean_f) / 2

        tc = compute_temporal_consistency(pred_masks_all)
        area_mean, area_max, area_exceed = compute_area_stability(pred_masks_all)

        result = {
            'model': model_name,
            'video': video,
            'J': round(mean_j * 100, 2),
            'F': round(mean_f * 100, 2),
            'J&F': round(jf * 100, 2),
            'temporal_consistency': round(tc * 100, 2),
            'area_change_mean': round(area_mean * 100, 2),
            'area_change_max': round(area_max * 100, 2),
            'area_exceed_20pct': area_exceed,
            'num_frames': len(gt_frames),
            'num_pred_frames': len(pred_frames),
        }
        results.append(result)
        print(f"  {video}: J={result['J']:.1f} F={result['F']:.1f} "
              f"J&F={result['J&F']:.1f} TC={result['temporal_consistency']:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='DAVIS 2017 Segmentation Evaluation')
    parser.add_argument('--davis_root', type=str, default='/data/datasets/DAVIS',
                        help='Path to DAVIS dataset root')
    parser.add_argument('--output_dir', type=str, default='/data/liuluyan/davis2017_eval_output',
                        help='Output directory for results')
    args = parser.parse_args()

    gt_dir = os.path.join(args.davis_root, 'Annotations', '480p')
    jpeg_dir = os.path.join(args.davis_root, 'JPEGImages', '480p')
    val_txt = os.path.join(args.davis_root, 'ImageSets', '2017', 'val.txt')

    # Read val set video list
    with open(val_txt, 'r') as f:
        videos = [line.strip() for line in f if line.strip()]
    print(f"DAVIS 2017 val set: {len(videos)} videos\n")

    # Define models and their result directories
    models = {
        'DEVA': '/data/liuluyan/Tracking-Anything-with-DEVA/davis2017_results',
        'Cutie': '/data/liuluyan/Cutie/davis2017_results/Annotations',
        'GSAM2': '/data/liuluyan/Grounded-SAM-2-clean/davis2017_results',
        'XMem': '/data/liuluyan/XMem/davis2017_results',
        'SAM2': '/data/liuluyan/SAM2/davis2017_results',
    }

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for model_name, model_dir in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"  Dir: {model_dir}")
        print(f"{'='*60}")
        results = evaluate_model(model_name, model_dir, gt_dir, jpeg_dir, videos)
        all_results.extend(results)

    # ---- Summary Table ----
    print(f"\n\n{'='*80}")
    print("SUMMARY: Per-Model Averages")
    print(f"{'='*80}")
    print(f"{'Model':<10} {'J':>8} {'F':>8} {'J&F':>8} {'TC':>8} "
          f"{'ΔArea%':>8} {'Exceed':>8}")
    print("-" * 70)

    summary = {}
    for model_name in models:
        model_results = [r for r in all_results if r['model'] == model_name]
        if not model_results:
            continue
        avg_j = np.mean([r['J'] for r in model_results])
        avg_f = np.mean([r['F'] for r in model_results])
        avg_jf = np.mean([r['J&F'] for r in model_results])
        avg_tc = np.mean([r['temporal_consistency'] for r in model_results])
        avg_area = np.mean([r['area_change_mean'] for r in model_results])
        total_exceed = sum([r['area_exceed_20pct'] for r in model_results])

        print(f"{model_name:<10} {avg_j:>8.2f} {avg_f:>8.2f} {avg_jf:>8.2f} "
              f"{avg_tc:>8.2f} {avg_area:>8.2f} {total_exceed:>8d}")

        summary[model_name] = {
            'J': round(avg_j, 2),
            'F': round(avg_f, 2),
            'J&F': round(avg_jf, 2),
            'temporal_consistency': round(avg_tc, 2),
            'area_change_mean_pct': round(avg_area, 2),
            'total_exceed_20pct_frames': total_exceed,
            'num_videos': len(model_results),
        }

    # ---- Save results ----
    # 1. Detailed JSON
    output_json = os.path.join(args.output_dir, 'eval_detailed.json')
    with open(output_json, 'w') as f:
        json.dump({'summary': summary, 'per_video': all_results}, f, indent=2)
    print(f"\nDetailed results saved to: {output_json}")

    # 2. CSV
    output_csv = os.path.join(args.output_dir, 'eval_per_video.csv')
    if all_results:
        keys = all_results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
    print(f"Per-video CSV saved to: {output_csv}")

    # 3. Summary CSV
    output_summary = os.path.join(args.output_dir, 'eval_summary.csv')
    with open(output_summary, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'J', 'F', 'J&F', 'Temporal_Consistency',
                         'Area_Change_Mean_Pct', 'Exceed_20pct_Frames', 'Num_Videos'])
        for model_name, s in summary.items():
            writer.writerow([model_name, s['J'], s['F'], s['J&F'],
                             s['temporal_consistency'], s['area_change_mean_pct'],
                             s['total_exceed_20pct_frames'], s['num_videos']])
    print(f"Summary CSV saved to: {output_summary}")


if __name__ == '__main__':
    main()
