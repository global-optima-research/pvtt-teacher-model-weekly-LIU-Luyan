"""
DAVIS 2017 J&F Evaluation Script
Computes J (region similarity / IoU) and F (contour accuracy) metrics
for video object segmentation results.
"""
import os
import sys
import numpy as np
from PIL import Image
import cv2
import json
from collections import defaultdict

# DAVIS 2017 val set path
GT_DIR = '/data/datasets/DAVIS/Annotations/480p'
VAL_LIST = '/data/datasets/DAVIS/ImageSets/2017/val.txt'

# Model results directories
MODELS = {
    'SAM2': '/data/liuluyan/SAM2/davis2017_results',
    'GSAM2': '/data/liuluyan/Grounded-SAM-2-clean/davis2017_results',
    'Cutie': '/data/liuluyan/Cutie/davis2017_results/Annotations',
    'DEVA': '/data/liuluyan/Tracking-Anything-with-DEVA/davis2017_results',
    'XMem': '/data/liuluyan/XMem/davis2017_results',
}


def load_mask(path):
    """Load a segmentation mask as numpy array."""
    mask = np.array(Image.open(path))
    return mask


def db_eval_iou(annotation, segmentation):
    """Compute region similarity (Jaccard index / IoU) for each object."""
    obj_ids = np.unique(annotation)
    obj_ids = obj_ids[obj_ids > 0]

    if len(obj_ids) == 0:
        return 1.0 if np.sum(segmentation > 0) == 0 else 0.0

    j_scores = []
    for obj_id in obj_ids:
        gt = (annotation == obj_id)
        pred = (segmentation == obj_id)
        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        if union == 0:
            j_scores.append(1.0)
        else:
            j_scores.append(intersection / union)
    return np.mean(j_scores)


def db_eval_boundary(annotation, segmentation, bound_th=0.008):
    """Compute contour accuracy (F-measure) for each object."""
    obj_ids = np.unique(annotation)
    obj_ids = obj_ids[obj_ids > 0]

    if len(obj_ids) == 0:
        return 1.0 if np.sum(segmentation > 0) == 0 else 0.0

    f_scores = []
    for obj_id in obj_ids:
        gt = (annotation == obj_id).astype(np.uint8)
        pred = (segmentation == obj_id).astype(np.uint8)

        # Get contours
        gt_boundary = _get_boundary(gt, bound_th)
        pred_boundary = _get_boundary(pred, bound_th)

        if gt_boundary.sum() == 0 and pred_boundary.sum() == 0:
            f_scores.append(1.0)
            continue
        if gt_boundary.sum() == 0 or pred_boundary.sum() == 0:
            f_scores.append(0.0)
            continue

        # Compute precision and recall
        bound_pix = max(1, round(bound_th * np.sqrt(gt.shape[0] * gt.shape[1])))

        # For each predicted boundary pixel, find if there's a gt boundary pixel nearby
        from scipy.ndimage import distance_transform_edt
        gt_dil = distance_transform_edt(1 - gt_boundary) <= bound_pix
        pred_dil = distance_transform_edt(1 - pred_boundary) <= bound_pix

        precision = np.sum(pred_boundary & gt_dil) / max(1, np.sum(pred_boundary))
        recall = np.sum(gt_boundary & pred_dil) / max(1, np.sum(gt_boundary))

        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            f_scores.append(2 * precision * recall / (precision + recall))

    return np.mean(f_scores)


def _get_boundary(mask, th=0.008):
    """Extract boundary from binary mask using morphological operations."""
    if mask.sum() == 0:
        return mask
    kernel_size = max(1, round(th * np.sqrt(mask.shape[0] * mask.shape[1])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel)
    boundary = mask - eroded
    return boundary


def find_pred_dir(model_dir, video_name):
    """Find the prediction directory for a given video, handling nested structures."""
    # Direct path
    direct = os.path.join(model_dir, video_name)
    if os.path.isdir(direct):
        return direct
    return None


def evaluate_model(model_name, pred_base_dir, videos):
    """Evaluate a model on all videos."""
    j_per_video = {}
    f_per_video = {}

    for video in videos:
        gt_video_dir = os.path.join(GT_DIR, video)
        pred_video_dir = find_pred_dir(pred_base_dir, video)

        if pred_video_dir is None:
            print(f'  WARNING: {model_name} - no predictions for {video}')
            continue

        gt_frames = sorted([f for f in os.listdir(gt_video_dir) if f.endswith('.png')])
        pred_frames = sorted([f for f in os.listdir(pred_video_dir) if f.endswith('.png')])

        if len(pred_frames) == 0:
            print(f'  WARNING: {model_name} - no prediction frames for {video}')
            continue

        j_scores = []
        f_scores = []

        for frame_file in gt_frames:
            gt_path = os.path.join(gt_video_dir, frame_file)
            pred_path = os.path.join(pred_video_dir, frame_file)

            gt_mask = load_mask(gt_path)

            if os.path.exists(pred_path):
                pred_mask = load_mask(pred_path)
                # Resize pred if needed
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = np.array(Image.fromarray(pred_mask).resize(
                        (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST))
            else:
                pred_mask = np.zeros_like(gt_mask)

            j = db_eval_iou(gt_mask, pred_mask)
            f = db_eval_boundary(gt_mask, pred_mask)
            j_scores.append(j)
            f_scores.append(f)

        j_per_video[video] = np.mean(j_scores)
        f_per_video[video] = np.mean(f_scores)

    return j_per_video, f_per_video


def main():
    # Load val video list
    with open(VAL_LIST, 'r') as f:
        videos = [line.strip() for line in f if line.strip()]
    print(f'Evaluating on {len(videos)} DAVIS 2017 val videos\n')

    all_results = {}

    for model_name, pred_dir in MODELS.items():
        print(f'Evaluating {model_name}...')
        if not os.path.exists(pred_dir):
            print(f'  Directory not found: {pred_dir}')
            continue

        j_per_video, f_per_video = evaluate_model(model_name, pred_dir, videos)

        if len(j_per_video) == 0:
            print(f'  No results for {model_name}')
            continue

        j_mean = np.mean(list(j_per_video.values()))
        f_mean = np.mean(list(f_per_video.values()))
        jf_mean = (j_mean + f_mean) / 2

        all_results[model_name] = {
            'J_mean': float(j_mean),
            'F_mean': float(f_mean),
            'J&F': float(jf_mean),
            'J_per_video': {k: float(v) for k, v in j_per_video.items()},
            'F_per_video': {k: float(v) for k, v in f_per_video.items()},
        }

        print(f'  J={j_mean:.4f}  F={f_mean:.4f}  J&F={jf_mean:.4f}  ({len(j_per_video)} videos)')

    # Summary table
    print('\n' + '=' * 60)
    print(f'{"Model":<12} {"J&F":>8} {"J":>8} {"F":>8}')
    print('-' * 60)
    for model_name in MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            print(f'{model_name:<12} {r["J&F"]:>8.4f} {r["J_mean"]:>8.4f} {r["F_mean"]:>8.4f}')
    print('=' * 60)

    # Per-video comparison
    print('\n\nPer-video J&F scores:')
    print(f'{"Video":<30}', end='')
    for model_name in MODELS:
        if model_name in all_results:
            print(f'{model_name:>10}', end='')
    print()
    print('-' * (30 + 10 * len(all_results)))

    for video in videos:
        print(f'{video:<30}', end='')
        for model_name in MODELS:
            if model_name in all_results:
                j = all_results[model_name]['J_per_video'].get(video, 0)
                f = all_results[model_name]['F_per_video'].get(video, 0)
                jf = (j + f) / 2
                print(f'{jf:>10.4f}', end='')
        print()

    # Find best/worst cases
    print('\n\nBest/Worst cases per model:')
    for model_name in MODELS:
        if model_name not in all_results:
            continue
        r = all_results[model_name]
        jf_scores = {v: (r['J_per_video'][v] + r['F_per_video'][v]) / 2
                     for v in r['J_per_video']}
        sorted_videos = sorted(jf_scores.items(), key=lambda x: x[1])
        worst3 = sorted_videos[:3]
        best3 = sorted_videos[-3:]
        print(f'\n{model_name}:')
        print(f'  Best:  {", ".join(f"{v}({s:.3f})" for v, s in reversed(best3))}')
        print(f'  Worst: {", ".join(f"{v}({s:.3f})" for v, s in worst3)}')

    # Save results to JSON
    output_path = '/data/liuluyan/repo/davis2017_eval_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    main()
