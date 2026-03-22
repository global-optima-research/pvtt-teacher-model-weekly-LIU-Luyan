import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

JPEG_DIR = "/data/datasets/DAVIS/JPEGImages/480p"
MODEL_DIRS = {
    "DEVA": "/data/liuluyan/Tracking-Anything-with-DEVA/davis2017_results",
    "Cutie": "/data/liuluyan/Cutie/davis2017_results/Annotations",
    "GSAM2": "/data/liuluyan/Grounded-SAM-2-clean/davis2017_results",
    "XMem": "/data/liuluyan/XMem/davis2017_results",
    "SAM2": "/data/liuluyan/SAM2/davis2017_results",
}
OUTPUT_DIR = "/data/liuluyan/davis2017_comparison"

COLORS = [
    (0, 0, 0),
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (255, 128, 0),
]

def colorize_mask(mask_img):
    mask_arr = np.array(mask_img)
    h, w = mask_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for uid in np.unique(mask_arr):
        if uid == 0:
            continue
        rgb[mask_arr == uid] = COLORS[uid % len(COLORS)]
    return Image.fromarray(rgb)

def overlay_mask(orig, mask_colored, alpha=0.5):
    mask_arr = np.array(mask_colored)
    orig_arr = np.array(orig).copy()
    nonzero = mask_arr.sum(axis=2) > 0
    orig_arr[nonzero] = (orig_arr[nonzero] * (1 - alpha) + mask_arr[nonzero] * alpha).astype(np.uint8)
    return Image.fromarray(orig_arr)

def add_label(img, label, font_size=24):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = img.width - tw - 10
    y = 5
    draw.rectangle([x - 5, y - 2, x + tw + 5, y + th + 4], fill=(0, 0, 0))
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return img

os.makedirs(OUTPUT_DIR, exist_ok=True)

videos = sorted(os.listdir(MODEL_DIRS["DEVA"]))
print(f"Processing {len(videos)} videos...")

count = 0
for video in videos:
    jpeg_dir = os.path.join(JPEG_DIR, video)
    if not os.path.isdir(jpeg_dir):
        print(f"  Skipping {video}: no JPEG dir")
        continue

    frames = sorted([f for f in os.listdir(jpeg_dir) if f.endswith(".jpg")])
    n = len(frames)
    if n == 0:
        continue

    # First, middle, last
    indices = [0, n // 2, n - 1]
    # Deduplicate in case n<=2
    indices = sorted(set(indices))

    for fidx in indices:
        frame_jpg = f"{fidx:05d}.jpg"
        frame_png = f"{fidx:05d}.png"

        orig_path = os.path.join(jpeg_dir, frame_jpg)
        if not os.path.exists(orig_path):
            continue
        orig = Image.open(orig_path).convert("RGB")

        orig_labeled = orig.copy()
        add_label(orig_labeled, "Original", font_size=24)

        panels = [orig_labeled]
        for model_name, model_dir in MODEL_DIRS.items():
            mask_path = os.path.join(model_dir, video, frame_png)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask_colored = colorize_mask(mask)
                overlaid = overlay_mask(orig.copy(), mask_colored, alpha=0.5)
                add_label(overlaid, model_name, font_size=24)
                panels.append(overlaid)
            else:
                blank = Image.new("RGB", orig.size, (50, 50, 50))
                add_label(blank, f"{model_name} (N/A)", font_size=24)
                panels.append(blank)

        total_w = sum(p.width for p in panels)
        h = panels[0].height
        concat = Image.new("RGB", (total_w, h))
        x_off = 0
        for p in panels:
            concat.paste(p, (x_off, 0))
            x_off += p.width

        out_name = f"{video}_frame{fidx:05d}.png"
        concat.save(os.path.join(OUTPUT_DIR, out_name))
        count += 1
        print(f"  Saved: {out_name}")

print(f"\nDone! Generated {count} comparison images.")
