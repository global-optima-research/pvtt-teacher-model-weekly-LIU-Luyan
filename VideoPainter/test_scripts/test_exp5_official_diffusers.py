"""Exp5: Pure CogVideoX I2V with OFFICIAL diffusers (not custom fork).
If grid: hardware/PyTorch issue. If no grid: custom fork is buggy."""
import os, sys, torch, cv2, numpy as np
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# IMPORTANT: Do NOT import custom fork. Remove it from path if present.
custom_paths = [p for p in sys.path if 'VideoPainter/diffusers' in p]
for p in custom_paths:
    sys.path.remove(p)

OUT = "/data/liuluyan/VideoPainter/test_v5_official_diffusers_i2v"
os.makedirs(OUT, exist_ok=True)

# Check which diffusers we're using
import diffusers
print(f"diffusers version: {diffusers.__version__}")
print(f"diffusers location: {diffusers.__file__}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video

# Input image
from decord import VideoReader
vr = VideoReader("/data/liuluyan/VideoPainter/test_videos/bunny.mp4")
first_frame = cv2.resize(vr[0].asnumpy(), (720, 480))
input_image = Image.fromarray(first_frame).convert("RGB")

print("Loading official CogVideoX-5b-I2V pipeline...")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "/data/liuluyan/VideoPainter/ckpt/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
pipe.scheduler = CogVideoXDPMScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing")
pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload()

print("Running pure I2V (13 frames, 50 steps)...")
result = pipe(
    prompt="A cute bunny plush toy on a table",
    image=input_image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=13,
    use_dynamic_cfg=True,
    guidance_scale=6.0,
    generator=torch.Generator().manual_seed(42),
    output_type="np"
).frames[0]

export_to_video(list(result), f"{OUT}/result.mp4", fps=8)
for idx, name in [(0, "first"), (len(result)//2, "mid"), (len(result)-1, "last")]:
    cv2.imwrite(f"{OUT}/frame_{name}.png",
        cv2.cvtColor((result[idx]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

mid = result[len(result)//2]
even = mid[::2, :, :].astype(float)
odd = mid[1::2, :, :].astype(float)
mn = min(len(even), len(odd))
diff = np.abs(even[:mn] - odd[:mn]).mean()
print(f"Even-odd row diff: {diff:.4f}")

has_grid = diff > 0.03

report = f"""# Experiment 5: Official diffusers CogVideoX I2V

## Purpose
Test CogVideoX I2V with OFFICIAL diffusers (not custom VideoPainter fork).
This isolates whether the grid is from the custom fork or from PyTorch/hardware.

## Setup
- diffusers: {diffusers.__version__} ({diffusers.__file__})
- PyTorch: {torch.__version__}
- GPU: {torch.cuda.get_device_name(0)}
- Pure I2V, no inpainting, no branch
- 13 frames, 50 steps

## Results
- Even-odd row diff: {diff:.4f}
- Grid present: {"YES" if has_grid else "NO"}

## Conclusion
"""
if has_grid:
    report += """Grid appears with OFFICIAL diffusers too!
Root cause: PyTorch {torch_ver} + RTX 5090 Blackwell CUDA kernel issue.
NOT a custom fork bug.
The original LED_grid_artifact_analysis.md was CORRECT about the hardware cause.
The Colab A100 result was a false lead (Colab also used PyTorch 2.10.0).

## Solution
Must use a non-Blackwell GPU (A100/H100/4090) with a standard PyTorch version.
""".format(torch_ver=torch.__version__)
else:
    report += """Grid does NOT appear with official diffusers!
Root cause: The custom VideoPainter diffusers fork (0.31.0.dev0) has a bug in its
CogVideoX implementation that causes the grid.

## Solution
Fix the custom diffusers fork, or use official diffusers for the base model.
"""

with open(f"{OUT}/report.md", "w") as f:
    f.write(report)
print(f"Report: {OUT}/report.md")
print(f"Done! -> {OUT}/")
