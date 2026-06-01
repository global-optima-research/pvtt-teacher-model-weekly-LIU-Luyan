#!/usr/bin/env python3
"""Generate 4 thesis figures for SAM3 PVTT full reproduction (PDF + PNG, dpi=300)."""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import image as mpimg
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
FULL_ROOT = Path("/data/liuluyan/sam3/sam3_pvtt_full")
IOU_SUMMARY_CSV = FULL_ROOT / "iou_summary.csv"
META_DIR = FULL_ROOT / "meta"
ANALYSIS_STATS = FULL_ROOT / "analysis_stats.json"
OVERLAYS_ROOT = FULL_ROOT / "overlays"

CATEGORY_RE = re.compile(r"\d+-([a-zA-Z]+)\d*")
DPI = 300
HIST_COLOR = "#4C72B0"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  saved {path}")


def load_frame_table() -> pd.DataFrame:
    """Per-frame rows from meta/*.json (761 frames)."""
    rows = []
    for meta_path in sorted(META_DIR.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = meta.get("video", meta_path.stem)
        category = meta.get("category") or parse_category(video)
        scene = meta.get("scene")
        has_scene = "_scene" in video
        for fr in meta.get("frames", []):
            rows.append(
                {
                    "video": video,
                    "frame_idx": int(fr["frame_idx"]),
                    "iou": float(fr["iou"]),
                    "category": category,
                    "has_scene": has_scene,
                    "shot_group": "Short single-shot clips"
                    if has_scene
                    else "Uncut full-length videos",
                }
            )
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} frame-level IoU records from meta/*.json")
    return df


def parse_category(name: str) -> str:
    m = CATEGORY_RE.search(name)
    return m.group(1).lower() if m else "unknown"


def load_video_summary() -> pd.DataFrame:
    df = pd.read_csv(IOU_SUMMARY_CSV)
    print(f"iou_summary.csv columns: {df.columns.tolist()}")
    print(f"iou_summary.csv rows (videos): {len(df)}")
    return df


def lookup_frame_iou(df_frames: pd.DataFrame, video: str, frame_idx: int) -> float | None:
    sub = df_frames[(df_frames["video"] == video) & (df_frames["frame_idx"] == frame_idx)]
    if sub.empty:
        return None
    return float(sub.iloc[0]["iou"])


def resolve_overlay(path: Path, video_dir: Path | None = None) -> Path | None:
    if path.is_file():
        return path
    if video_dir and video_dir.is_dir():
        candidates = sorted(video_dir.glob("frame_*.jpg"))
        if candidates:
            print(f"  warning: fallback {path} -> {candidates[0]}")
            return candidates[0]
    return None


def plot_iou_hist(df_frames: pd.DataFrame) -> None:
    print("Plotting sam3_iou_hist...")
    ious = df_frames["iou"].to_numpy()
    mean_v = float(np.mean(ious))
    median_v = float(np.median(ious))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ious, bins=30, color=HIST_COLOR, edgecolor="white", linewidth=0.4)
    ax.axvline(mean_v, color="red", linestyle="--", linewidth=1.5, label=f"mean = {mean_v:.3f}")
    ax.axvline(
        median_v,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"median = {median_v:.3f}",
    )
    ax.set_xlabel("IoU")
    ax.set_ylabel("Number of frames")
    ax.legend(loc="upper left")
    sns.despine(ax=ax)
    save_figure(fig, "sam3_iou_hist")
    plt.close(fig)


def plot_category_bar(df_frames: pd.DataFrame) -> None:
    print("Plotting sam3_category_bar...")
    grp = df_frames.groupby("category")["iou"]
    stats = grp.agg(["count", "mean"]).reset_index()
    stats = stats.sort_values("mean", ascending=True)

    print("Category (count, mean IoU):")
    for _, row in stats.iterrows():
        print(f"  {row['category']}: count={int(row['count'])}, mean={row['mean']:.3f}")

    n = len(stats)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, n))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(stats["category"], stats["mean"], color=colors, edgecolor="none")
    ax.set_xlabel("Mean IoU")
    ax.set_ylabel("")
    ax.set_xlim(0, max(1.0, stats["mean"].max() * 1.12))

    for bar, val in zip(bars, stats["mean"]):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    sns.despine(ax=ax)
    save_figure(fig, "sam3_category_bar")
    plt.close(fig)


def plot_shot_box(df_frames: pd.DataFrame) -> None:
    print("Plotting sam3_shot_box...")
    order = ["Uncut full-length videos", "Short single-shot clips"]
    groups = {g: df_frames.loc[df_frames["shot_group"] == g, "iou"].to_numpy() for g in order}
    n_full = len(groups[order[0]])
    n_short = len(groups[order[1]])
    print(f"  Uncut full-length videos: n={n_full}, mean={groups[order[0]].mean():.3f}")
    print(f"  Short single-shot clips: n={n_short}, mean={groups[order[1]].mean():.3f}")
    if not (200 <= n_full <= 300):
        print(f"  WARNING: full-length count {n_full} (expected ~247)")
    if not (450 <= n_short <= 550):
        print(f"  WARNING: short-clip count {n_short} (expected ~514)")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    plot_df = df_frames[["shot_group", "iou"]].copy()
    sns.boxplot(
        data=plot_df,
        x="shot_group",
        y="iou",
        order=order,
        width=0.5,
        color="#DDDDDD",
        fliersize=2,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="shot_group",
        y="iou",
        order=order,
        color=HIST_COLOR,
        alpha=0.25,
        size=2,
        jitter=0.25,
        ax=ax,
    )

    for i, g in enumerate(order):
        vals = groups[g]
        if len(vals) == 0:
            continue
        mean_v = float(np.mean(vals))
        ax.text(
            i,
            1.02,
            f"n={len(vals)}\nmean={mean_v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_ylabel("IoU")
    ax.set_xlabel("")
    ax.set_ylim(-0.05, 1.08)
    sns.despine(ax=ax)
    save_figure(fig, "sam3_shot_box")
    plt.close(fig)


def plot_qualitative(df_frames: pd.DataFrame) -> None:
    print("Plotting sam3_qualitative...")
    panels = [
        {
            "key": "a",
            "title_tag": "high",
            "label": "Watch",
            "video": "0047-watch7_scene02",
            "frame": 3,
            "path": OVERLAYS_ROOT / "0047-watch7_scene02" / "frame_03.jpg",
            "fallback_dir": OVERLAYS_ROOT / "0047-watch7_scene02",
        },
        {
            "key": "b",
            "title_tag": "medium",
            "label": "Bracelet",
            "video": "0024-bracelet6_scene02",
            "frame": 3,
            "path": OVERLAYS_ROOT / "0024-bracelet6_scene02" / "frame_03.jpg",
            "fallback_dir": OVERLAYS_ROOT / "0024-bracelet6_scene02",
        },
        {
            "key": "c",
            "title_tag": "low",
            "label": "Necklace",
            "video": "0034-necklace1",
            "frame": 1,
            "path": OVERLAYS_ROOT / "0034-necklace1" / "frame_01.jpg",
            "fallback_dir": OVERLAYS_ROOT / "0034-necklace1",
        },
        {
            "key": "d",
            "title_tag": "lowest",
            "label": "Earring",
            "video": "0031-earring3_scene04",
            "frame": None,
            "path": OVERLAYS_ROOT / "0031-earring3_scene04" / "frame_00.jpg",
            "fallback_dir": OVERLAYS_ROOT / "0031-earring3_scene04",
        },
    ]

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.08, hspace=0.12)

    for idx, p in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        img_path = resolve_overlay(p["path"], p["fallback_dir"])
        if img_path is None:
            print(f"  warning: missing overlay for {p['video']}, leaving blank")
            ax.text(0.5, 0.5, "Missing", ha="center", va="center")
        else:
            ax.imshow(mpimg.imread(img_path))

        iou_val = None
        if p["frame"] is not None:
            iou_val = lookup_frame_iou(df_frames, p["video"], p["frame"])
        if iou_val is None and p["fallback_dir"] and p["fallback_dir"].is_dir():
            # try frame_00 if earring
            for cand in sorted(p["fallback_dir"].glob("frame_*.jpg")):
                stem = cand.stem  # frame_00
                try:
                    fi = int(stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
                iou_val = lookup_frame_iou(df_frames, p["video"], fi)
                if iou_val is not None:
                    break

        if iou_val is not None:
            subtitle = f"({p['key']}) {p['label']} — IoU={iou_val:.3f} ({p['title_tag']})"
        else:
            subtitle = f"({p['key']}) {p['label']} ({p['title_tag']})"

        ax.set_title(subtitle, fontsize=11, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    save_figure(fig, "sam3_qualitative")
    plt.close(fig)


def main() -> None:
    print(f"Output directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _ = load_video_summary()
    df_frames = load_frame_table()

    plotters = [
        ("sam3_iou_hist", lambda: plot_iou_hist(df_frames)),
        ("sam3_category_bar", lambda: plot_category_bar(df_frames)),
        ("sam3_shot_box", lambda: plot_shot_box(df_frames)),
        ("sam3_qualitative", lambda: plot_qualitative(df_frames)),
    ]

    for name, fn in plotters:
        try:
            fn()
        except Exception as e:
            print(f"ERROR plotting {name}: {e}")
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()
