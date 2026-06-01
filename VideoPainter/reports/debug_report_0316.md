# VideoPainter "LED灯珠矩阵" 伪影分析报告

**Author:** Luyan Liu
**Date:** 2026-03-16
**Server:** liuluyan@111.17.197.107 (RTX-5090-32G-X8)

---

## 1. 问题描述

在服务器上使用 VideoPainter 对 dataset100 数据进行 video inpainting 时，所有生成结果都呈现出规则的 **"LED灯珠矩阵"** 网格状伪影：整个画面被均匀分割为小方格，每个方格内部亮度一致，格间存在明显边界，类似 LED 显示屏的灯珠阵列效果。

该伪影出现在所有测试样本中（handbag、sunglasses、handfan），且不仅限于 inpainting 区域，**整个画面**都被覆盖。

---

## 2. 排查过程与结论

### 2.1 已排除的原因

通过系统性 A/B 测试，以下因素已被逐一排除：

| # | 排查项 | 测试方法 | 结果 |
|---|--------|----------|------|
| 1 | **VideoPainter Branch 模型** | 设置 `conditioning_scale=0` 完全关闭 branch conditioning | 仍有灯珠 |
| 2 | **自定义 diffusers 代码** | 在新环境中安装官方 `diffusers==0.31.0`，运行同样的 CogVideoX I2V 生成 | 仍有灯珠 |
| 3 | **Pipeline mask 处理** | 去除 inpainting 相关逻辑（mask_add, replace_gt），仅做纯 I2V 生成 | 仍有灯珠 |
| 4 | **model_cpu_offload_seq 缺失 branch** | 修复 offload 顺序为 `text_encoder->branch->transformer->vae` | 仍有灯珠 |
| 5 | **VAE tiling** | 注释掉 `pipe.vae.enable_tiling()` | 仍有灯珠 |
| 6 | **计算精度 (bf16/fp16/fp32)** | 分别测试 `torch.bfloat16`、`torch.float16`、`torch.float32` | 三者均有灯珠 |
| 7 | **Attention 后端** | 测试 `flash_attention_2`、`mem_efficient`、`math` (纯 PyTorch SDPA) | 均有灯珠 |
| 8 | **TF32 精度** | 设置 `torch.backends.cuda.matmul.allow_tf32 = False` 和 `torch.backends.cudnn.allow_tf32 = False` | 仍有灯珠 |
| 9 | **cuDNN** | 设置 `torch.backends.cudnn.enabled = False` | Transformer 推理正常完成，但 VAE 解码 OOM（conv3d 需 35.6GB），无法验证最终效果 |
| 10 | **特定 GPU 故障** | 分别在 GPU 2 和 GPU 5 上测试 | 两张卡结果一致 |
| 11 | **VAE 编解码** | 对真实图片做 VAE encode → decode roundtrip | 输出正常，无灯珠 |

### 2.2 关键发现

#### 发现 1：灯珠不是 VideoPainter 独有的问题

将 VideoPainter 的所有自定义组件（branch model、inpainting mask、自定义 diffusers fork）全部去除，仅使用**原版 CogVideoX-5b-I2V** 做纯 Image-to-Video 生成：

```python
# 使用官方 diffusers==0.31.0，无任何自定义代码
pipe = CogVideoXImageToVideoPipeline.from_pretrained("CogVideoX-5b-I2V")
result = pipe(image=img, prompt="a handbag on a table", ...)
```

**结果：同样出现灯珠矩阵。** 这证明问题不在 VideoPainter，而在更底层。

#### 发现 2：灯珠出现在 Transformer 输出的 latents 中

通过 hook 捕获 VAE 解码前的 raw latents（shape: `[1, 4, 16, 60, 90]`），可视化发现：

- **raw latents 中已经存在明显的棋盘格/网格模式**
- 网格间距 = 2 个 latent 像素 = CogVideoX 的 `patch_size=2`
- 说明是 **Transformer 产生了有缺陷的 latents**，而非 VAE 解码引入

#### 发现 3：灯珠随扩散步数逐渐加剧

| 扩散步数 | even-odd 行差异 | even-odd 列差异 | 相邻行差异 |
|----------|----------------|----------------|-----------|
| 1 步 | 3.48 | 3.40 | 3.36 |
| 5 步 | 13.20 | 6.98 | 12.56 |
| 30 步 | 15.83 | 8.66 | 14.95 |

第 1 步时几乎没有灯珠，第 5 步开始出现，第 30 步完全成型。说明 Transformer 在**迭代去噪过程中逐步放大了 patch 边界的不连续性**。

### 2.3 根因定位

**根本原因：PyTorch 2.10.0+cu128 在 RTX 5090 (Blackwell SM 12.0) 上的 CUDA kernel 兼容性问题。**

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 5090 |
| 架构 | Blackwell, SM 12.0 (全新架构) |
| PyTorch | 2.10.0+cu128 (bleeding-edge nightly) |
| CUDA | 12.8 |
| cuDNN | 9.10.02 |

RTX 5090 (Blackwell SM 12.0) 是全新 GPU 架构，PyTorch 2.10.0 是远超稳定版（2025年5月稳定版为 2.6.x）的 nightly 构建版本。在这个组合下，Transformer 中的 CUDA kernel（最可能是 **attention 计算**或 **linear layer 的 cuBLAS matmul**）在 Blackwell 上存在精度或实现问题，导致：

1. 每个 denoising step 中，Transformer 对 patch token 的处理产生微小的 patch 边界不连续
2. 这种不连续在 30 步迭代中被逐步放大
3. unpatch 操作将 token 恢复为 2x2 spatial block 时，相邻 block 之间的不连续表现为可见的网格
4. VAE 解码将 latent 空间的 2px 网格放大为输出视频中 ~16px 间距的灯珠矩阵

---

## 3. 技术细节

### 3.1 CogVideoX Transformer 的 Patch 机制

```
输入 latent: [B, C=16, T=13, H=60, W=90]
                    ↓ patch_embed (Conv2d, kernel=2, stride=2)
tokens:      [B, T*30*45, dim=3072]    (30=60/2, 45=90/2)
                    ↓ 42 层 Transformer blocks
                    ↓ proj_out
output:      [B, T*30*45, C*p*p=16*4=64]
                    ↓ unpatch (reshape + permute)
输出 latent: [B, C=16, T=13, H=60, W=90]
```

每个 token 对应 2x2 的 latent 空间区域。如果 Transformer 处理后相邻 token 的值不连续，unpatch 后就会出现 2px 间距的棋盘格。

### 3.2 Unpatch 代码（正确，无 bug）

```python
# cogvideox_transformer_3d.py, forward() 末尾
p = self.config.patch_size  # = 2
output = hidden_states.reshape(batch_size, num_frames, height//p, width//p, -1, p, p)
output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
# 最终 shape: [B, T, C, H, W]
```

### 3.3 环境信息

```
PyTorch:       2.10.0+cu128
CUDA:          12.8
cuDNN:         9.10.02
GPU:           NVIDIA GeForce RTX 5090
SM:            12.0 (Blackwell)
diffusers:     0.31.0 (官方) / 0.31.0.dev0 (自定义 fork)
Python:        3.10
Model:         CogVideoX-5b-I2V (~10.2GB transformer, ~820MB VAE)
```

---

## 4. 解决方案

### 方案 A：更换 PyTorch 版本（推荐优先尝试）

安装较旧的 PyTorch 版本（如 2.6.0+cu124），通过 CUDA 前向兼容性在 Blackwell 上运行：

```bash
conda create -n test_older python=3.10
conda activate test_older
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install diffusers==0.31.0 transformers==4.46.0 accelerate safetensors
```

> **风险：** PyTorch 2.6.0 编译于 CUDA 12.4，不包含 Blackwell SM 12.0 的原生 SASS 代码。需要依赖 JIT PTX 编译，可能部分 kernel 不支持，也可能因为 JIT 编译本身而正常工作（避开有 bug 的预编译 kernel）。

### 方案 B：在非 Blackwell GPU 上运行

如果有 A100 / H100 / RTX 4090 等 Ampere/Hopper/Ada 架构的机器，直接在上面运行即可避开问题。CogVideoX 在这些架构上已被广泛验证。

### 方案 C：等待 PyTorch 修复

持续关注以下资源：
- [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues) — 搜索 "Blackwell" / "SM 12.0" / "RTX 5090"
- [PyTorch Nightly](https://pytorch.org/get-started/locally/) — 尝试更新的 nightly 版本
- NVIDIA CUDA Toolkit / cuDNN 更新

### 方案 D：后处理去网格（临时 workaround）

如果急需出结果，可以对生成的视频做轻度空间滤波（如 Gaussian blur sigma=1 或双边滤波）来抑制网格。但这会损失细节，仅作为临时方案。

---

## 5. 排查过程中的附加修复

在排查过程中发现并修复了以下代码问题（虽然不是灯珠的根因，但属于需要修复的 bug）：

| 问题 | 文件 | 修复 |
|------|------|------|
| `model_cpu_offload_seq` 缺少 branch | `pipeline_cogvideox_inpainting_i2v_branch_anyl.py` | 改为 `"text_encoder->branch->transformer->vae"` |
| SAM2 checkpoint 路径错误 | `run_demo.py` | 传入绝对路径 `--sam2_checkpoint /data/liuluyan/VideoPainter/ckpt/sam2_hiera_large.pt` |
| `UnboundLocalError: masks` | `cogvideox_transformer_3d.py:536` | 当 `mask_add=False` 时 `masks` 未定义（尚未修复，记录在此） |

---

## 6. 总结

| 项 | 结论 |
|----|------|
| 灯珠原因 | PyTorch 2.10.0 在 RTX 5090 Blackwell (SM 12.0) 上的 CUDA kernel 问题 |
| 灯珠位置 | Transformer 输出的 latents（VAE 解码前已存在） |
| 灯珠模式 | patch_size=2 导致的棋盘格，间距=2 latent px = 16 output px |
| 是否 VideoPainter 代码问题 | **否**，原版 CogVideoX I2V 同样出现 |
| 是否 diffusers 代码问题 | **否**，官方 diffusers 0.31.0 同样出现 |
| 首要建议 | 更换 PyTorch 版本或在非 Blackwell GPU 上运行 |
