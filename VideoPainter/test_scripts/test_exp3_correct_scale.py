"""Exp3: Correct A/B test - pass conditioning_scale to pipe() call, not branch attribute"""
import os, sys, gc, torch, cv2, numpy as np
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, "/data/liuluyan/VideoPainter/app")

# SAM2
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
gt_mask0 = mask_pil[0]
mask_pil[0] = Image.fromarray(np.zeros_like(np.array(mask_pil[0]))).convert("RGB")
gt_vid0 = video_pil[0]

from diffusers import CogVideoXDPMScheduler, CogvideoXBranchModel, CogVideoXI2VDualInpaintAnyLPipeline
from diffusers.utils import export_to_video


def run_test(name, cond_scale):
    OUT = f"/data/liuluyan/VideoPainter/test_v3_{name}"
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Running: {name} (conditioning_scale={cond_scale} passed to pipe())")
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

    local_mask = list(mask_pil)
    local_mask[0] = Image.fromarray(np.zeros_like(np.array(mask_pil[0]))).convert("RGB")

    print(f"Running inference ({num_frames} frames)...")
    out = pipe(prompt="A cute bunny plush toy on a table", image=masked_pil[0],
        num_videos_per_prompt=1, num_inference_steps=50, num_frames=num_frames,
        use_dynamic_cfg=True, guidance_scale=6.0, generator=torch.Generator().manual_seed(42),
        video=masked_pil, masks=local_mask, strength=1.0,
        replace_gt=True, mask_add=True, stride=num_frames, prev_clip_weight=0.0,
        conditioning_scale=cond_scale,  # <-- CORRECTLY passed to pipe()
        output_type="np").frames[0]

    export_to_video(list(out), f"{OUT}/result.mp4", fps=8)
    for idx, fname in [(0, "first"), (len(out)//2, "mid"), (len(out)-1, "last")]:
        cv2.imwrite(f"{OUT}/frame_{fname}.png",
            cv2.cvtColor((out[idx]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Grid analysis
    mid = out[len(out)//2]
    even = mid[::2, :, :].astype(float)
    odd = mid[1::2, :, :].astype(float)
    mn = min(len(even), len(odd))
    diff = np.abs(even[:mn] - odd[:mn]).mean()
    print(f"Even-odd row diff: {diff:.4f}")

    report = f"""# Experiment 3: {name}

## Purpose
Correctly pass conditioning_scale={cond_scale} to pipe() call (not branch attribute).

## Setup
- conditioning_scale={cond_scale} passed to pipe.__call__()
- enable_model_cpu_offload, offload_seq includes branch
- VAE slicing: yes, tiling: no
- {num_frames} frames, 50 steps

## Results
- Even-odd row diff: {diff:.4f}
- Grid present: {"YES" if diff > 0.03 else "NO"}

## Analysis
"""
    if cond_scale == 0.0:
        report += "Branch completely disabled via conditioning_scale=0.\n"
        report += "If grid gone: branch causes grid. If grid remains: base model issue.\n"
    else:
        report += "Standard run with full branch conditioning.\n"

    with open(f"{OUT}/report.md", "w") as f:
        f.write(report)

    del pipe, branch; gc.collect(); torch.cuda.empty_cache()
    print(f"Saved to {OUT}/")
    return diff


d1 = run_test("branch_scale1", cond_scale=1.0)
d2 = run_test("branch_scale0", cond_scale=0.0)

summary = f"""# Experiment 3 Summary: Branch A/B Test (Correct)

| Run | conditioning_scale | Even-Odd Diff | Grid? |
|-----|-------------------|---------------|-------|
| branch_scale1 | 1.0 | {d1:.4f} | {"YES" if d1 > 0.03 else "NO"} |
| branch_scale0 | 0.0 | {d2:.4f} | {"YES" if d2 > 0.03 else "NO"} |

## Conclusion
"""
if d1 > 0.03 and d2 > 0.03:
    summary += "Grid present in BOTH → Issue is in base CogVideoX / pipeline, NOT branch.\n"
    summary += "Next: test pure CogVideoX I2V without any inpainting pipeline.\n"
elif d1 > 0.03 and d2 <= 0.03:
    summary += "Grid ONLY with branch → Branch conditioning causes grid.\n"
    summary += "Next: investigate branch model weights or branch-transformer interaction.\n"
else:
    summary += "No grid in either → previous grid was from something else (tiling/slicing?).\n"

with open("/data/liuluyan/VideoPainter/test_v3_summary.md", "w") as f:
    f.write(summary)
print(f"\n{'='*60}")
print("FINAL SUMMARY:")
print(summary)
