"""Download VPData remaining files using wget (more robust than huggingface_hub)."""
import json, subprocess, os

MIRROR = "https://hf-mirror.com"
REPO = "datasets/TencentARC/VPData"
BASE_DIR = "/data/liuluyan/data/VPData"

def list_files(folder):
    """List files in a folder via HF API with pagination."""
    all_files = []
    cursor = ""
    while True:
        url = f"{MIRROR}/api/{REPO}/tree/main/{folder}?recursive=false&limit=1000"
        if cursor:
            url += f"&cursor={cursor}"
        r = subprocess.run(["curl", "-s", url], capture_output=True, text=True)
        data = json.loads(r.stdout)
        if not isinstance(data, list) or len(data) == 0:
            break
        all_files.extend(data)
        if len(data) < 1000:
            break
        # Build cursor from last item for next page
        import base64
        last = data[-1]["path"]
        cursor_obj = json.dumps({"file_name": last, "tree_oid": ""})
        cursor = base64.b64encode(cursor_obj.encode()).decode()
    return all_files

def download_file(path, size):
    """Download a single LFS file via wget with resume support."""
    url = f"{MIRROR}/{REPO}/resolve/main/{path}"
    local_path = os.path.join(BASE_DIR, path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Skip if already downloaded and size matches
    if os.path.exists(local_path):
        local_size = os.path.getsize(local_path)
        if local_size == size:
            print(f"  SKIP (complete): {path}")
            return True
        elif local_size > 0:
            print(f"  RESUME: {path} ({local_size/1e9:.2f}/{size/1e9:.2f} GB)")

    print(f"  Downloading: {path} ({size/1e9:.2f} GB)")
    ret = subprocess.run(
        ["wget", "-c", "-q", "--show-progress", "-O", local_path, url],
        timeout=7200,  # 2 hour timeout per file
    )
    return ret.returncode == 0

folders = ["pexels_masks", "videovo_masks", "videovo_raw_videos"]
# Also check small files
small_files = [
    "pexels.csv", "videovo.csv",
    "pexels_videovo_test_dataset.csv", "pexels_videovo_val_dataset.csv",
]

for sf in small_files:
    local_path = os.path.join(BASE_DIR, sf)
    if not os.path.exists(local_path):
        url = f"{MIRROR}/{REPO}/resolve/main/{sf}"
        print(f"Downloading {sf}...")
        subprocess.run(["wget", "-c", "-q", "-O", local_path, url])

for folder in folders:
    print(f"\n=== {folder} ===")
    try:
        files = list_files(folder)
    except Exception as e:
        print(f"  Error listing: {e}")
        continue

    total = len(files)
    total_size = sum(f.get("size", 0) for f in files)
    print(f"  {total} files, {total_size/1e9:.1f} GB total")

    done = 0
    failed = []
    for i, f in enumerate(files):
        path = f["path"]
        size = f.get("size", 0)
        try:
            ok = download_file(path, size)
            if ok:
                done += 1
            else:
                failed.append(path)
        except Exception as e:
            print(f"  FAILED: {path} - {e}")
            failed.append(path)

    print(f"  Completed: {done}/{total}")
    if failed:
        print(f"  Failed: {failed}")

print("\nDone!")
