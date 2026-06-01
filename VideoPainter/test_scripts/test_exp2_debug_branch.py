"""Exp2: Compare branch ON vs branch OFF (conditioning_scale=0) to verify branch effect"""
import os, sys, gc, torch, cv2, numpy as np
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, "/data/liuluyan/VideoPainter/app")

# === SAM2 (shared for both runs) ===
from decord import VideoReader
vr = VideoReader("test_videos/bunny.mp4")
fps_orig = vr.get_avg_fps()
step = max(1, int(fps_orig / 8))
indices = list(range(0, min(int(3.2 * fps_orig), len(vr)), step))
frames = np.array([cv2.resize(f, (720, 480)) for f in vr.get_batch(indices).asnumpy()])

from sam2.build_sam import build_sam2_video_predictor
predictor = build_sam2_video_predictor("sam2_hiera_l.yaml", "ckpt/sam2_hiera_large.pt")
state = predictor.init_state(images=frames, offload_video_to_cpu=True, async_loading_frames=True)
predictor.reset_state(state)
predictor.add_new_points(inference_state=state, frame_idx=0, obj_id=0,
    points=np.array([[360, 240]], dtype=np.float32), labels=np.array([1], dtype=np.int32))
masks = np.zeros((len(frames), 480, 720), dtype=np.uint8)
for fi, _, ml in predictor.propagate_in_video(state):
    masks[fi] = (ml[0, 0] > 0).cpu().numpy().astype(np.uint8)
del predictor, state; gc.collect(); torch.cuda.empty_cache()

# Prepare data
video_pil, masked_pil, mask_pil = [], [], []
for i in range(len(frames)):
    fr, mk = frames[i], masks[i]
    dl = cv2.dilate(mk, np.ones((32, 32), np.uint8))
    video_pil.append(Image.fromarray(fr).convert("RGB"))
    mf = fr.copy(); mf[dl > 0] = 0
    masked_pil.append(Image.fromarray(mf).convert("RGB"))
    mr = np.zeros_like(fr); mr[dl > 0] = 255
    mask_pil.append(Image.fromarray(mr.astype(np.uint8)).convert("RGB"))

n = min(len(video_pil), 49)
num_frames = ((n - 1) // 4) * 4 + 1
video_pil = video_pil[:num_frames]
masked_pil = masked_pil[:num_frames]
mask_pil = mask_pil[:num_frames]
gt_mask0, gt_vid0 = mask_pil[0], video_pil[0]
mask_pil[0] = Image.fromarray(np.zeros_like(np.array(mask_pil[0]))).convert("RGB")

from diffusers import CogVideoXDPMScheduler, CogvideoXBranchModel, CogVideoXI2VDualInpaintAnyLPipeline
from diffusers.utils import export_to_video


def run_experiment(name, conditioning_scale, mask_add):
    OUT = f"/data/liuluyan/VideoPainter/test_v2_{name}"
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Running: {name} (conditioning_scale={conditioning_scale}, mask_add={mask_add})")
    print(f"{'='*60}")

    branch = CogvideoXBranchModel.from_pretrained("ckpt/VideoPainter/checkpoints/branch",
        torch_dtype=torch.bfloat16).to(torch.bfloat16)
    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained("ckpt/CogVideoX-5b-I2V",
        branch=branch, torch_dtype=torch.bfloat16)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_slicing()
    pipe.enable_model_cpu_offload()

    # Set conditioning scale on branch
    pipe.branch.conditioning_scale = conditioning_scale

    # Reset mask for first frame
    local_mask_pil = list(mask_pil)
    local_mask_pil[0] = Image.fromarray(np.zeros_like(np.array(mask_pil[0]))).convert("RGB")

    print(f"offload seq: {pipe.model_cpu_offload_seq}")
    print(f"branch conditioning_scale: {pipe.branch.conditioning_scale}")
    print(f"Running inference ({num_frames} frames)...")

    out = pipe(prompt="A cute bunny plush toy on a table", image=masked_pil[0],
        num_videos_per_prompt=1, num_inference_steps=50, num_frames=num_frames,
        use_dynamic_cfg=True, guidance_scale=6.0, generator=torch.Generator().manual_seed(42),
        video=masked_pil, masks=local_mask_pil, strength=1.0,
        replace_gt=True, mask_add=mask_add, stride=num_frames, prev_clip_weight=0.0,
        output_type="np").frames[0]

    export_to_video(list(out), f"{OUT}/result.mp4", fps=8)
    for idx, fname in [(0, "first"), (len(out)//2, "mid"), (len(out)-1, "last")]:
        cv2.imwrite(f"{OUT}/frame_{fname}.png",
            cv2.cvtColor((out[idx]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Check for grid pattern: compare even/odd rows in mid frame
    mid = out[len(out)//2]
    even_rows = mid[::2, :, :]
    odd_rows = mid[1::2, :, :]
    min_len = min(len(even_rows), len(odd_rows))
    row_diff = np.abs(even_rows[:min_len].astype(float) - odd_rows[:min_len].astype(float)).mean()
    print(f"Even-odd row diff (grid indicator): {row_diff:.4f}")

    has_grid = "YES (grid detected)" if row_diff > 0.05 else "NO (looks clean)"

    report = f"""# Experiment: {name}

## Setup
- conditioning_scale: {conditioning_scale}
- mask_add: {mask_add}
- enable_model_cpu_offload with offload_seq: {pipe.model_cpu_offload_seq}
- VAE slicing: yes, tiling: no
- {num_frames} frames, 50 steps

## Results
- Even-odd row diff (grid indicator): {row_diff:.4f}
- LED Grid present: {has_grid}

## Analysis
"""
    if conditioning_scale == 0:
        report += """This run completely disables branch conditioning (scale=0).
If grid STILL appears: branch is NOT the cause of the grid.
If grid DISAPPEARS: branch conditioning is somehow creating the grid.
"""
    elif not mask_add:
        report += """This run disables mask_add, so branch conditioning is applied to ALL regions
(not just non-masked regions). This changes how the inpainted area is treated.
"""
    else:
        report += """Standard run with branch enabled and mask_add=True.
"""

    with open(f"{OUT}/report.md", "w") as f:
        f.write(report)

    del pipe, branch; gc.collect(); torch.cuda.empty_cache()
    print(f"Saved to {OUT}/")
    return row_diff


# Run experiments
diff_branch_on = run_experiment("branch_on_mask_add", conditioning_scale=1.0, mask_add=True)
diff_branch_off = run_experiment("branch_off_scale0", conditioning_scale=0.0, mask_add=True)
diff_no_mask_add = run_experiment("branch_on_no_mask_add", conditioning_scale=1.0, mask_add=False)

# Summary report
summary = f"""# Debug Summary: Branch Effect on LED Grid

## Results
| Experiment | Conditioning Scale | mask_add | Even-Odd Row Diff | Grid? |
|---|---|---|---|---|
| branch_on_mask_add | 1.0 | True | {diff_branch_on:.4f} | {"YES" if diff_branch_on > 0.05 else "NO"} |
| branch_off_scale0 | 0.0 | True | {diff_branch_off:.4f} | {"YES" if diff_branch_off > 0.05 else "NO"} |
| branch_on_no_mask_add | 1.0 | False | {diff_no_mask_add:.4f} | {"YES" if diff_no_mask_add > 0.05 else "NO"} |

## Conclusion
"""
if diff_branch_on > 0.05 and diff_branch_off > 0.05:
    summary += "Grid appears regardless of branch → Branch is NOT the cause. Issue is in base CogVideoX or pipeline.\n"
elif diff_branch_on > 0.05 and diff_branch_off <= 0.05:
    summary += "Grid only with branch ON → Branch conditioning is CAUSING the grid.\n"
else:
    summary += "Unexpected results, need further investigation.\n"

with open("/data/liuluyan/VideoPainter/test_v2_debug_summary.md", "w") as f:
    f.write(summary)
print(f"\n{'='*60}")
print("SUMMARY:")
print(summary)
