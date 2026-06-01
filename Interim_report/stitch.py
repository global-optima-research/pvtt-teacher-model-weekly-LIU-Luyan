# -*- coding: utf-8 -*-
"""
拼接 DAVIS 2017 多模型对比图
每张原图是 1x7 的横排：Original | GT | DEVA | Cutie | GSAM2 | XMem | SAM2
"""

import os
from PIL import Image, ImageDraw, ImageFont

# ============ 配置区 ============
ROOT = r"davis2017_comparison"

# 想要拼进去的视频（按顺序），改这里就行
SELECTED_VIDEOS = [
    "paragliding-launch_frame00040.png",
    "kite-surf_frame00025.png",
    "lab-coat_frame00023.png",
    "bmx-trees_frame00021.png",
    "gold-fish_frame00039.png",
]

# 原图列顺序（必须和图里的顺序一致）
COL_NAMES_ALL = ["Original", "GT", "DEVA", "Cutie", "GSAM2", "XMem", "SAM2"]

# 想保留哪些列（按这里给的顺序输出）。想全保留就写 COL_NAMES_ALL
KEEP_COLS = ["Original", "GT", "DEVA", "XMem", "SAM2"]

# 输出图片尺寸控制
TARGET_ROW_WIDTH = 2400      # 每行最终宽度（像素），太大文件巨大
GAP = 6                      # 行之间的间距
OUTPUT_NAME = "D:\Learning_file\master&PhD\master\HKUST\毕设\IP-2026-spring\pvtt-teacher-model-weekly-LIU-Luyan\Interim_report\davis_qualitative.png"
# ===============================


def crop_columns(img: Image.Image, all_cols, keep_cols):
    """把 1x7 大图按列均匀切，只保留 keep_cols 中的列，再水平拼回去"""
    n = len(all_cols)
    W, H = img.size
    col_w = W // n  # 每列宽度

    keep_idx = [all_cols.index(c) for c in keep_cols]
    pieces = []
    for i in keep_idx:
        left = i * col_w
        right = (i + 1) * col_w if i < n - 1 else W  # 最后一列吃掉余数
        pieces.append(img.crop((left, 0, right, H)))

    total_w = sum(p.width for p in pieces)
    out = Image.new("RGB", (total_w, H), (255, 255, 255))
    x = 0
    for p in pieces:
        out.paste(p, (x, 0))
        x += p.width
    return out


def resize_to_width(img: Image.Image, target_w: int):
    ratio = target_w / img.width
    new_h = int(round(img.height * ratio))
    return img.resize((target_w, new_h), Image.LANCZOS)


def add_column_headers(img: Image.Image, col_names, header_h=40):
    """在最上方加一行列标题"""
    W, H = img.size
    n = len(col_names)
    col_w = W / n
    out = Image.new("RGB", (W, H + header_h), (255, 255, 255))
    out.paste(img, (0, header_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", size=int(header_h * 0.55))
    except OSError:
        font = ImageFont.load_default()
    for i, name in enumerate(col_names):
        bbox = draw.textbbox((0, 0), name, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = i * col_w + col_w / 2 - tw / 2
        cy = (header_h - th) / 2 - bbox[1]
        draw.text((cx, cy), name, fill=(0, 0, 0), font=font)
    return out


def main():
    rows = []
    for fname in SELECTED_VIDEOS:
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            print(f"[skip] not found: {path}")
            continue
        img = Image.open(path).convert("RGB")
        img = crop_columns(img, COL_NAMES_ALL, KEEP_COLS)
        img = resize_to_width(img, TARGET_ROW_WIDTH)
        rows.append(img)

    if not rows:
        print("没有可用图片")
        return

    total_h = sum(r.height for r in rows) + GAP * (len(rows) - 1)
    canvas = Image.new("RGB", (TARGET_ROW_WIDTH, total_h), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + GAP

    # 顶部加列标题
    canvas = add_column_headers(canvas, KEEP_COLS, header_h=46)

    out_path = os.path.join(ROOT, OUTPUT_NAME)
    canvas.save(out_path, quality=92)
    print(f"saved -> {out_path}")
    print(f"size  = {canvas.size}")


if __name__ == "__main__":
    main()