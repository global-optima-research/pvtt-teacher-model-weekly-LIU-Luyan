"""Exp8: Exact replica of Gradio demo (app/utils.py) inference logic.
Uses load_model() from utils.py + FLUX first-frame inpainting."""
import os, sys, gc, torch, cv2, numpy as np
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.insert(0, "/data/liuluyan/VideoPainter/app")

OUT = "/data/liuluyan/VideoPainter/test_v8_gradio_logic"
os.makedirs(OUT, exist_ok=True)

# ============ SAM2 ============
print("=== Step 1: SAM2 ===")
from decord import VideoReader
vr = VideoReader("/data/liuluyan/VideoPainter/test_videos/bunny.mp4")
fps_orig = vr.get_avg_fps()
step = max(1, int(fps_orig / 8))
indices = list(range(0, min(int(3.2 * fps_orig), len(vr)), step))
all_frames = np.array([cv2.resize(f, (720, 480)) for f in vr.get_batch(indices).asnumpy()])
print(f"Extracted {len(all_frames)} frames")

from sam2.build_sam import build_sam2_video_predictor
predictor = build_sam2_video_predictor("sam2_hiera_l.yaml",
    "/data/liuluyan/VideoPainter/ckpt/sam2_hiera_large.pt")
state = predictor.init_state(images=all_frames, offload_video_to_cpu=True, async_loading_frames=True)
predictor.reset_state(state)
predictor.add_new_points(inference_state=state, frame_idx=0, obj_id=0,
    points=np.array([[360, 240]], dtype=np.float32),
    labels=np.array([1], dtype=np.int32))
raw_masks = np.zeros((len(all_frames), 480, 720), dtype=np.uint8)
for fi, _, ml in predictor.propagate_in_video(state):
    raw_masks[fi] = (ml[0, 0] > 0).cpu().numpy().astype(np.uint8)
del predictor, state; gc.collect(); torch.cuda.empty_cache()
print(f"SAM2 done, masks: {raw_masks.shape}")

# ============ Prepare data (Gradio style) ============
# Gradio demo: images = list of RGB PIL, masks = list of RGB PIL (white=inpaint)
# Use up to 49 frames
max_f = min(len(all_frames), 49)
num_frames = ((max_f - 1) // 4) * 4 + 1
print(f"Using {num_frames} frames")

images = []  # masked video (object blacked out)
masks = []   # binary masks (white = inpaint region)
originals = []  # original frames for visualization
for i in range(num_frames):
    fr = all_frames[i]
    mk = raw_masks[i]
    originals.append(Image.fromarray(fr).convert("RGB"))
    # Create masked frame (black out object) - Gradio passes masked frames as "images"
    mf = fr.copy()
    mf[mk > 0] = 0
    images.append(Image.fromarray(mf).convert("RGB"))
    # Binary mask
    bm = np.where(mk > 0, 255, 0).astype(np.uint8)
    masks.append(Image.fromarray(bm).convert("RGB"))

# Save visualization
f0 = all_frames[0].copy()
cv2.circle(f0, (360, 240), 8, (0, 255, 0), -1)
ov = f0.copy(); ov[raw_masks[0] > 0] = [255, 0, 0]
blend = cv2.addWeighted(f0, 0.6, ov, 0.4, 0)
cv2.imwrite(f"{OUT}/frame0_with_mask.png", cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))

# ============ Load models (EXACT Gradio load_model()) ============
print("\n=== Step 2: Load models (Gradio style) ===")
from diffusers import (CogVideoXDPMScheduler, CogvideoXBranchModel,
    CogVideoXI2VDualInpaintAnyLPipeline, CogVideoXTransformer3DModel, FluxFillPipeline)
from diffusers.utils import export_to_video

dtype = torch.bfloat16
device = "cuda:0"
model_path = "/data/liuluyan/VideoPainter/ckpt/CogVideoX-5b-I2V"
branch_path = "/data/liuluyan/VideoPainter/ckpt/VideoPainter/checkpoints/branch"
id_adapter = "/data/liuluyan/VideoPainter/ckpt/VideoPainterID/checkpoints"
flux_path = "/data/liuluyan/VideoPainter/ckpt/flux_inp"

# Exact copy of load_model() from utils.py
branch = CogvideoXBranchModel.from_pretrained(branch_path, torch_dtype=dtype).to(device, dtype=dtype)

transformer = CogVideoXTransformer3DModel.from_pretrained(
    model_path, subfolder="transformer", torch_dtype=dtype,
    id_pool_resample_learnable=True).to(device, dtype=dtype)

pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
    model_path, branch=branch, transformer=transformer, torch_dtype=dtype)

pipe.load_lora_weights(id_adapter,
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="test_1", target_modules=["transformer"])
print(f"Adapters: {pipe.get_list_adapters()}")

pipe.text_encoder.requires_grad_(False)
pipe.transformer.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.branch.requires_grad_(False)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.to(device)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
print(f"Pipeline GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Load FLUX to CPU (Gradio style)
try:
    pipe_img = FluxFillPipeline.from_pretrained(flux_path, torch_dtype=dtype)
    print("FLUX loaded (on CPU)")
except Exception as e:
    pipe_img = None
    print(f"FLUX not available: {e}")

# ============ Run inference (EXACT Gradio run_video_inpainting()) ============
print("\n=== Step 3: Inference (Gradio style) ===")
prompt = "A cute bunny plush toy on a table"
image_inpainting_prompt = "A stuffed animal toy"
dilate_size = 16  # Gradio default

# Dilate masks (Gradio style)
print(f"Dilating masks with size {dilate_size}...")
for i in range(len(masks)):
    mask = cv2.dilate(np.array(masks[i]), np.ones((dilate_size, dilate_size)))
    masks[i] = Image.fromarray(mask.astype(np.uint8))

# FLUX first-frame inpainting (Gradio style)
if pipe_img is not None:
    print("FLUX first-frame inpainting...")
    pipe_img.to("cuda")
    image_inpainting = pipe_img(
        prompt=image_inpainting_prompt,
        image=originals[0],  # original first frame
        mask_image=masks[0],
        height=originals[0].size[1],
        width=originals[0].size[0],
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]
    pipe_img.to("cpu")
    torch.cuda.empty_cache()
    images[0] = image_inpainting
    image_inpainting.save(f"{OUT}/first_frame_flux_inpainted.png")
    masks[0] = Image.fromarray(np.zeros_like(np.array(masks[0]))).convert("RGB")
    print("FLUX done!")
else:
    print("No FLUX, using original first frame")
    masks[0] = Image.fromarray(np.zeros_like(np.array(masks[0]))).convert("RGB")

# Run VideoPainter (EXACT params from utils.py)
print(f"Running VideoPainter ({num_frames} frames, 50 steps)...")
inpaint_outputs = pipe(
    prompt=prompt,
    image=images[0],
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=num_frames,
    use_dynamic_cfg=True,
    guidance_scale=6.0,
    generator=torch.Generator().manual_seed(42),
    video=images,
    masks=masks,
    strength=1.0,
    replace_gt=True,
    mask_add=True,
    stride=num_frames,
    prev_clip_weight=0.0,
    id_pool_resample_learnable=False,  # Gradio demo uses False!
    output_type="np"
).frames[0]

print(f"Output: {inpaint_outputs.shape}")

# ============ Save results ============
print("\n=== Step 4: Save ===")
export_to_video(list(inpaint_outputs), f"{OUT}/result.mp4", fps=8)
for idx, name in [(0, "first"), (len(inpaint_outputs)//2, "mid"), (len(inpaint_outputs)-1, "last")]:
    cv2.imwrite(f"{OUT}/frame_{name}.png",
        cv2.cvtColor((inpaint_outputs[idx]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# Grid check
mid = inpaint_outputs[len(inpaint_outputs)//2]
even = mid[::2, :, :].astype(float)
odd = mid[1::2, :, :].astype(float)
mn = min(len(even), len(odd))
diff = np.abs(even[:mn] - odd[:mn]).mean()
has_grid = "YES" if diff > 0.03 else "NO"
print(f"Even-odd diff: {diff:.4f}, Grid: {has_grid}")

report = f"""# Experiment 8: Gradio Demo Logic (Exact Replica)

## Purpose
Replicate EXACTLY the Gradio demo's inference logic from app/utils.py.

## What's different from previous experiments
1. Transformer loaded with id_pool_resample_learnable=True (Gradio style)
2. LoRA weights loaded from VideoPainterID
3. FLUX first-frame inpainting (if available)
4. id_pool_resample_learnable=False passed to pipe() (Gradio style)
5. dilate_size=16 (Gradio default, not 32)
6. pipe.to("cuda") — NO CPU offload

## Setup
- PyTorch: {torch.__version__}
- GPU: {torch.cuda.get_device_name(0)}
- {num_frames} frames, 50 steps
- FLUX first-frame: {"Yes" if pipe_img is not None else "No"}
- VAE slicing + tiling (for 32GB GPU)

## Results
- Even-odd diff: {diff:.4f}
- Grid present: {has_grid}

## Conclusion
{"Grid STILL present — issue is fundamental to this GPU/PyTorch combo" if has_grid == "YES" else "NO GRID! The missing components (FLUX first-frame + correct params) were the fix!"}
"""
with open(f"{OUT}/report.md", "w") as f:
    f.write(report)

print(f"\nDone! -> {OUT}/")
