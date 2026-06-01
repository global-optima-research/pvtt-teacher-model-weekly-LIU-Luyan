import json, subprocess, sys

base = "https://hf-mirror.com/api/datasets/TencentARC/VPData/tree/main"
for folder in ["pexels_masks", "videovo_masks", "videovo_raw_videos"]:
    url = f"{base}/{folder}?recursive=false&limit=2000"
    r = subprocess.run(["wget", "-q", url, "-O", "-"], capture_output=True, text=True)
    try:
        data = json.loads(r.stdout)
        total_size = sum(x.get("size", 0) for x in data)
        print(f"{folder}: {len(data)} files, {total_size/1e9:.1f} GB")
    except Exception as e:
        print(f"{folder}: error - {e}")
