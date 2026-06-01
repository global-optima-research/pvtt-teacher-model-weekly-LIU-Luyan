"""Exp6: pipe.to('cuda') + 13 frames + VAE slicing. NO CPU OFFLOAD AT ALL."""
import os, sys, gc, torch, cv2, numpy as np
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, "/data/liuluyan/VideoPainter/app")

OUT = "/data/liuluyan/VideoPainter/test_v6_no_offload_13frames"
os.makedirs(OUT, exist_ok=True)

# SAM2
from decord import VideoReader
vr = VideoReader("test_videos/bunny.mp4")
fps_orig = vr.get_avg_fps()
step = max(1, int(fps_orig / 8))
indices = list(range(0, min(int(3.2 * fps_orig), len(vr)), step))[:13]
frames = np.array([cv2.resize(f, (720, 480)) for f in vr.get_batch(indices).asnumpy()])
print(f"Frames: {len(frames)}")

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
    bm = np.where(dl > 0, 255, 0).astype(np.uint8)
    mask_pil.append(Image.fromarray(bm).convert("RGB"))

num_frames = 13  # 4*3+1
video_pil = video_pil[:num_frames]
masked_pil = masked_pil[:num_frames]
mask_pil = mask_pil[:num_frames]
gt_mask0, gt_vid0 = mask_pil[0], video_pil[0]
mask_pil[0] = Image.fromarray(np.zeros_like(np.array(mask_pil[0]))).convert("RGB")

# Load pipeline - OFFICIAL STYLE, NO OFFLOAD
from diffusers import CogVideoXDPMScheduler, CogvideoXBranchModel, CogVideoXI2VDualInpaintAnyLPipeline
from diffusers.utils import export_to_video

print("Loading branch to cuda...")
branch = CogvideoXBranchModel.from_pretrained("ckpt/VideoPainter/checkpoints/branch",
    torch_dtype=torch.bfloat16).to(torch.bfloat16).cuda()
print("Loading pipeline...")
pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained("ckpt/CogVideoX-5b-I2V",
    branch=branch, torch_dtype=torch.bfloat16)
pipe.text_encoder.requires_grad_(False)
pipe.transformer.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.branch.requires_grad_(False)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# OFFICIAL: pipe.to("cuda") + VAE slicing only (to avoid OOM on decode)
pipe.to("cuda")
pipe.vae.enable_slicing()
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

print(f"Running inference ({num_frames} frames, 50 steps) - NO CPU OFFLOAD...")
out = pipe(prompt="A cute bunny plush toy on a table", image=masked_pil[0],
    num_videos_per_prompt=1, num_inference_steps=50, num_frames=num_frames,
    use_dynamic_cfg=True, guidance_scale=6.0, generator=torch.Generator().manual_seed(42),
    video=masked_pil, masks=mask_pil, strength=1.0,
    replace_gt=True, mask_add=True, stride=num_frames, prev_clip_weight=0.0,
    output_type="np").frames[0]

# Save
mask_pil[0] = gt_mask0; video_pil[0] = gt_vid0
export_to_video(list(out), f"{OUT}/result.mp4", fps=8)
for idx, name in [(0, "first"), (len(out)//2, "mid"), (len(out)-1, "last")]:
    cv2.imwrite(f"{OUT}/frame_{name}.png",
        cv2.cvtColor((out[idx]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# Grid check
mid = out[len(out)//2]
even = mid[::2, :, :].astype(float)
odd = mid[1::2, :, :].astype(float)
mn = min(len(even), len(odd))
diff = np.abs(even[:mn] - odd[:mn]).mean()
print(f"Even-odd diff: {diff:.4f}")
has_grid = "YES" if diff > 0.03 else "NO"
print(f"Grid: {has_grid}")

report = f"""# Experiment 6: NO CPU OFFLOAD (pipe.to("cuda"))

## Purpose
Test with pipe.to("cuda") and NO cpu offload of any kind.
This matches exactly how the official code runs.

## Setup
- pipe.to("cuda") — all models on GPU simultaneously
- VAE slicing only (for decode memory)
- 13 frames (reduced to fit in 32GB)
- PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}
- NO enable_model_cpu_offload, NO enable_sequential_cpu_offload

## Results
- Even-odd diff: {diff:.4f}
- Grid: {has_grid}

## Conclusion
If NO grid: CPU offload was the root cause all along.
If grid: the issue is something else entirely.
"""
with open(f"{OUT}/report.md", "w") as f:
    f.write(report)
print(f"Done! -> {OUT}/")
