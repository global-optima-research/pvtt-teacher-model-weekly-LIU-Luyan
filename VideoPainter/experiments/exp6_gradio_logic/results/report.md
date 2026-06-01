# Experiment 8: Gradio Demo Logic (Exact Replica)

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
- PyTorch: 2.10.0+cu128
- GPU: NVIDIA GeForce RTX 5090
- 9 frames, 50 steps
- FLUX first-frame: No
- VAE slicing + tiling (for 32GB GPU)

## Results
- Even-odd diff: 0.0171
- Grid present: NO

## Conclusion
NO GRID! The missing components (FLUX first-frame + correct params) were the fix!
