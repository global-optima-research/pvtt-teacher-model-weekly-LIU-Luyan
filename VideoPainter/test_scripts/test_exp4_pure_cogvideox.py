"""Exp4: Pure CogVideoX I2V - NO inpainting, NO branch, NO custom pipeline.
Tests if the grid comes from CogVideoX model itself or from the inpainting pipeline code."""
import os, sys, torch, cv2, numpy as np
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

OUT = "/data/liuluyan/VideoPainter/test_v4_pure_cogvideox_i2v"
os.makedirs(OUT, exist_ok=True)

# Use the CUSTOM diffusers fork's standard I2V pipeline (not inpainting)
sys.path.insert(0, "/data/liuluyan/VideoPainter/diffusers/src")
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video

# Load first frame from test video as the input image
from decord import VideoReader
vr = VideoReader("test_videos/bunny.mp4")
first_frame = vr[0].asnumpy()
first_frame = cv2.resize(first_frame, (720, 480))
input_image = Image.fromarray(first_frame).convert("RGB")
input_image.save(f"{OUT}/input_frame.png")

# Load pipeline - standard I2V, no branch, no inpainting
print("Loading CogVideoX-5b-I2V pipeline (standard, no inpainting)...")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "ckpt/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
pipe.scheduler = CogVideoXDPMScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing")
pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload()

print("Running pure I2V generation (13 frames, 50 steps)...")
result = pipe(
    prompt="A cute bunny plush toy on a table, indoor scene",
    image=input_image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=13,
    use_dynamic_cfg=True,
    guidance_scale=6.0,
    generator=torch.Generator().manual_seed(42),
    output_type="np"
).frames[0]

print(f"Output shape: {result.shape}")

# Save
export_to_video(list(result), f"{OUT}/result.mp4", fps=8)
for idx, name in [(0, "first"), (len(result)//2, "mid"), (len(result)-1, "last")]:
    cv2.imwrite(f"{OUT}/frame_{name}.png",
        cv2.cvtColor((result[idx]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# Grid analysis
mid = result[len(result)//2]
even = mid[::2, :, :].astype(float)
odd = mid[1::2, :, :].astype(float)
mn = min(len(even), len(odd))
diff = np.abs(even[:mn] - odd[:mn]).mean()
print(f"Even-odd row diff: {diff:.4f}")

# Check patch-level grid at various strides
for stride in [2, 4, 8, 16]:
    a = mid[::stride*2, :, :].astype(float)
    b = mid[stride::stride*2, :, :].astype(float)
    mn2 = min(len(a), len(b))
    d = np.abs(a[:mn2] - b[:mn2]).mean()
    print(f"  stride={stride}: diff={d:.4f}")

has_grid = diff > 0.03

report = f"""# Experiment 4: Pure CogVideoX I2V (No Inpainting)

## Purpose
Test if the LED grid comes from CogVideoX model itself or from the inpainting pipeline.
This uses CogVideoXImageToVideoPipeline (standard I2V, no branch, no masks).

## Setup
- Pipeline: CogVideoXImageToVideoPipeline (NOT inpainting pipeline)
- No branch model, no masks, no inpainting
- Custom diffusers fork 0.31.0.dev0
- PyTorch: {torch.__version__}
- GPU: RTX 5090
- enable_model_cpu_offload, VAE slicing
- 13 frames, 50 steps

## Results
- Even-odd row diff: {diff:.4f}
- Grid present: {"YES" if has_grid else "NO"}

## Analysis
"""
if has_grid:
    report += """Grid appears even in pure I2V without inpainting!
This means the issue is in the base CogVideoX model or the diffusers fork,
NOT in the inpainting pipeline or branch model.

## Next Steps
1. Test with OFFICIAL diffusers (pip install diffusers==0.31.0) instead of custom fork
2. Test with a different PyTorch version
3. The grid might be a known issue with CogVideoX on certain PyTorch/CUDA versions
"""
else:
    report += """Grid does NOT appear in pure I2V!
This means the issue IS specific to the inpainting pipeline code.

## Next Steps
1. Investigate the inpainting pipeline's latent processing
2. Check how masks and masked_video_latents interact with the denoising loop
3. Compare inpainting pipeline's denoising loop with standard I2V
"""

with open(f"{OUT}/report.md", "w") as f:
    f.write(report)
print(f"\nReport: {OUT}/report.md")
print(f"Done! -> {OUT}/")
