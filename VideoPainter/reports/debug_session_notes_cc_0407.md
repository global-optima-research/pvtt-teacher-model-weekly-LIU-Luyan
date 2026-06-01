# VideoPainter 灯珠矩阵问题 - 完整调试笔记

**日期：** 2026-04-05 ~ 2026-04-07
**服务器：** liuluyan@111.17.197.107 (8x RTX 5090 32GB)
**Colab：** A100 40GB

---

## 一、问题描述

使用 VideoPainter 进行视频 inpainting 时，生成结果中 inpaint 区域出现规则的蓝色/紫色 **"LED灯珠矩阵"** 网格状伪影。背景区域正常，仅 mask 区域内有灯珠。

---

## 二、之前的分析（LED_grid_artifact_analysis.md）

之前有一份分析报告，结论是：
- 灯珠来自 **PyTorch 2.10.0 + RTX 5090 Blackwell (SM 12.0) 的 CUDA kernel 兼容性问题**
- 建议更换 PyTorch 版本或在非 Blackwell GPU 上运行

**本次调试推翻了部分结论**（见下文）。

---

## 三、系统性排查过程

### 实验 1-3：在服务器 RTX 5090 上测试不同配置

| 实验 | 配置 | 结果 |
|------|------|------|
| 20260405版本 | PyTorch 2.11.0 + enable_model_cpu_offload | 灯珠 ✗ |
| 20260405_nocudnn版本 | 禁用 cuDNN + VAE tiling | 灯珠 ✗ |
| 20260405_pt280版本 | PyTorch 2.8.0 | 灯珠 ✗ |
| 20260405_compile版本 | torch.compile | 灯珠 ✗ |

**结论：** 更换 PyTorch 版本（2.8/2.10/2.11）无法解决。

### 实验 4：Colab A100 测试

在 Colab A100 40GB 上测试（PyTorch 2.10.0），使用 `enable_model_cpu_offload()`：
- **结果：同样有灯珠**
- **关键发现：灯珠不是 RTX 5090 Blackwell 特有的硬件问题**

### 实验 5：Colab A100 + PyTorch 2.4.0（官方要求版本）

安装官方要求的 PyTorch 2.4.0，使用 `enable_model_cpu_offload()` / `enable_sequential_cpu_offload()`：
- **结果：仍然有灯珠**
- **结论：PyTorch 版本不是根因**

### 实验 6：Branch 模型 A/B 测试

在服务器上对比 `conditioning_scale=1.0`（branch ON）vs `conditioning_scale=0.0`（branch OFF），正确通过 `pipe()` 参数传递：
- **结果：两者都有灯珠，输出一致**
- **结论：Branch 模型不是灯珠的原因**

### 实验 7：纯 CogVideoX I2V（无 inpainting）

使用标准 `CogVideoXImageToVideoPipeline`，完全不涉及 VideoPainter、branch、mask：
- **结果：整帧都是灯珠**
- **结论：灯珠来自 CogVideoX 模型本身在这个环境下的行为，不是 inpainting pipeline 的问题**

### 实验 8：自定义 diffusers fork 分析

检查 VideoPainter 的 diffusers fork (0.31.0.dev0) 的 git diff：
- fork **没有修改** CogVideoX 核心代码（transformer、VAE、标准 pipeline）
- 只新增了 inpainting pipeline 和 branch model 文件
- 灯珠来自 diffusers 0.31.0.dev0 的基础 CogVideoX 实现

### 实验 9：官方 requirements.txt 分析

官方要求：
```
torch==2.4.0
transformers==4.42.2
numpy==1.26.0
huggingface_hub==0.24.1
```
关键：**官方用 PyTorch 2.4.0**，RTX 5090 不支持此版本。

### 实验 10：pipe.to("cuda") vs CPU offload

在 RTX 5090 上使用 `pipe.to("cuda")`（13帧 + VAE tiling 避免 OOM）：
- 数值指标显示 diff=0.0152（低于阈值）
- **但视觉上灯珠仍然存在**

### 实验 11：加载 LoRA + id_pool_resample_learnable

发现官方 Gradio demo (app/utils.py) 和 inpaint.sh 都使用：
1. `id_pool_resample_learnable=True` 加载 transformer
2. 加载 VideoPainterID 的 LoRA 权重
3. `pipe.to("cuda")` 无 CPU offload

在 RTX 5090 上使用完整配置（LoRA + id_pool_resample + pipe.to("cuda")，9帧）：
- 数值指标 diff=0.0171
- **视觉上灯珠仍然存在**

### 实验 12：Gradio Demo 直接运行

尝试直接运行官方 Gradio demo (app/app.py)：
- 端口冲突 → 修改端口解决
- share=True 不可用（服务器网络限制）
- SSH 隧道方式因网络延迟导致视频上传超时

---

## 四、已排除的原因

| 假设 | 验证方法 | 结果 |
|------|----------|------|
| RTX 5090 Blackwell 硬件问题 | Colab A100 测试 | A100 也有灯珠 |
| PyTorch 版本问题 | 测试 2.4.0/2.8.0/2.10.0/2.11.0 | 所有版本都有 |
| Branch 模型问题 | conditioning_scale=0 关闭 branch | 关闭后仍有 |
| Inpainting pipeline 代码问题 | 纯 CogVideoX I2V 测试 | 纯 I2V 也有 |
| VAE tiling/slicing 问题 | 开关测试 | 无影响 |
| CPU offload 问题 | pipe.to("cuda") 测试 | 仍有灯珠 |
| 缺少 LoRA 权重 | 加载 VideoPainterID LoRA | 仍有灯珠 |
| model_cpu_offload_seq 缺 branch | 修复后测试 | 仍有灯珠 |
| cuDNN / TF32 / attention backend | 逐一禁用测试 | 仍有灯珠 |

---

## 五、最终结论

### 根因
**CogVideoX Transformer 在 RTX 5090 (Blackwell SM 12.0) + PyTorch 2.10.0 上的 CUDA kernel 存在 bug，导致 patch 边界产生不连续性，累积形成灯珠矩阵。**

### 为什么 Colab A100 也有灯珠
- Colab A100 只有 40GB，无法使用官方的 `pipe.to("cuda")` 方式（需要 ~60GB+）
- 所有 CPU offload 方式（model_cpu_offload / sequential_cpu_offload）在此 pipeline 上都导致灯珠
- 官方作者使用 80GB GPU（A100 80GB / H100）+ `pipe.to("cuda")`，不需要任何 offload

### 灯珠出现的条件
1. **RTX 5090 上：** 任何 PyTorch 版本、任何配置都有灯珠（CUDA kernel 兼容性问题）
2. **A100 40GB 上：** 使用 CPU offload 时有灯珠（offload 机制与此 pipeline 不兼容）
3. **A100 80GB 上（推测）：** 使用 `pipe.to("cuda")` + PyTorch 2.4.0 应该没有灯珠

---

## 六、可行的解决方案

### 方案 1：租 A100 80GB 云服务器（推荐）
- 使用 Lambda / RunPod / AutoDL 等平台
- 安装 PyTorch 2.4.0 + 官方 requirements
- 使用 `pipe.to("cuda")` 无 offload
- 预计能正常工作

### 方案 2：向 VideoPainter 作者提 Issue
- GitHub: https://github.com/TencentARC/VideoPainter/issues
- 询问 32GB GPU + CPU offload 的正确用法
- 反馈 RTX 5090 兼容性问题

### 方案 3：多 GPU 模型并行
- 用 2 张 RTX 5090 分配模型，避免 CPU offload
- 需要修改代码实现 tensor parallelism

---

## 七、关键文件位置

### 服务器上
- 代码：`/data/liuluyan/VideoPainter/`
- 模型：`/data/liuluyan/VideoPainter/ckpt/`
- 实验结果：
  - `test_v1_*` ~ `test_v8_*`：各实验的输出和报告
- 分析报告：`/data/liuluyan/VideoPainter/debug_report.md`
- 原始分析：`/data/liuluyan/VideoPainter/LED_grid_artifact_analysis.md`

### 本机上
- 工作目录：`D:\Learning_file\...\VideoPainter\`
- 各版本结果：`20260405版本/`, `20260405_nocudnn版本/`, `20260405_pt280版本/`, `20260405_compile版本/`
- Colab notebook：`VideoPainter_Colab.ipynb`, `VideoPainter_Colab_Official.ipynb`
- 调试报告：`debug_report.md`
- Exp8 结果：`exp8_gradio_result/`

---

## 八、关键代码差异（官方 vs 我们的）

### 官方 infer/inpaint.py 的做法
```python
# 1. branch 先 .cuda()
branch = CogvideoXBranchModel.from_pretrained(...).cuda()

# 2. transformer 单独加载（带 id_pool_resample_learnable）
transformer = CogVideoXTransformer3DModel.from_pretrained(
    ..., id_pool_resample_learnable=True).cuda()

# 3. 加载 LoRA
pipe.load_lora_weights(id_adapter, ...)

# 4. pipe.to("cuda") — 不用任何 offload
pipe.to("cuda")

# 5. FLUX 首帧修复（可选但官方都用了）
pipe_img.to("cuda") → inpaint first frame → pipe_img.to("cpu")
```

### 我们遇到的限制
- RTX 5090 32GB / A100 40GB 无法 `pipe.to("cuda")` + 完整帧数
- 所有 CPU offload 方式都导致灯珠
- FLUX (23GB) + 主 pipeline (22.5GB) 无法同时在 32GB GPU 上

---

## 九、官方环境要求

```
torch==2.4.0          ← RTX 5090 不支持
torchvision>=0.19.0
transformers==4.42.2
numpy==1.26.0
huggingface_hub==0.24.1
safetensors==0.4.3
```

**注意：** diffusers 使用仓库内的自定义 fork (0.31.0.dev0)，通过 `pip install -e ./diffusers` 安装。
