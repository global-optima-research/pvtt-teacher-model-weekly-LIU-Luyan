# VideoPainter LED 灯珠矩阵 - 系统性排查报告

**日期：** 2026-04-07
**排查环境：** RTX 5090 (服务器) + A100 (Colab)

---

## 实验结果汇总

| # | 实验 | GPU | 灯珠？ | 结论 |
|---|------|-----|--------|------|
| 1 | VideoPainter inpainting + cpu_offload | RTX 5090 | YES | - |
| 2 | VideoPainter inpainting + cpu_offload | A100 (Colab) | YES | 不是硬件问题 |
| 3 | branch conditioning_scale=1.0 vs 0.0 | RTX 5090 | 都有 | 不是 branch 的问题 |
| 4 | 纯 CogVideoX I2V（无 inpainting） | RTX 5090 | YES | 不是 inpainting pipeline 的问题 |
| 5 | 官方 diffusers 0.31.0 测试 | RTX 5090 | 未完成（依赖冲突） | - |

## 根因定位

### 已排除
- ~~RTX 5090 Blackwell 硬件问题~~ → A100 上也有
- ~~VideoPainter branch 模型~~ → conditioning_scale=0 结果完全一样
- ~~Inpainting pipeline 代码~~ → 纯 CogVideoX I2V 也有灯珠
- ~~VAE tiling/slicing~~ → 关闭后仍有
- ~~model_cpu_offload 缺少 branch~~ → 修复后仍有

### 确定原因
**灯珠来自 CogVideoX Transformer 在 diffusers 0.31.0.dev0 + PyTorch 2.10.0 下的基础行为。**

证据链：
1. 自定义 fork 的 git diff 显示**没有修改** CogVideoX 核心代码（transformer、VAE、标准 pipeline）
2. 纯 CogVideoX I2V（标准 pipeline，非 inpainting）产生整帧灯珠
3. fork 基于 **diffusers 0.31.0.dev0**（预发布版本），可能包含后来在正式版中修复的 bug
4. 服务器和 Colab 都使用 **PyTorch 2.10.0+cu128**

## 可能的解决方案

### 方案 A（推荐）：升级 diffusers 版本
将 VideoPainter 的自定义 diffusers fork 基于更新的 diffusers 版本（如 0.32.0+）重建。
具体步骤：
1. 克隆官方 diffusers 最新稳定版
2. 将 VideoPainter 新增的文件移植过去：
   - `models/branch_cogvideox.py`
   - `pipelines/cogvideo/pipeline_cogvideox_inpainting_i2v_branch_anyl.py`
   - transformer 中关于 branch_block_samples 的修改
3. 测试是否解决灯珠问题

### 方案 B：降级 PyTorch
在 A100 或其他 Ampere GPU 上使用 PyTorch 2.4/2.5 + 官方 diffusers 0.31.0 测试。

### 方案 C：使用官方推理脚本
VideoPainter 官方的 `infer/inpaint.py` 设计用于特定的数据格式（CSV + npz masks）。
如果官方演示（如 Gradio app）能正常工作，说明问题在环境配置。

## 下一步
1. 在 Colab 上安装官方 diffusers 0.32+，测试纯 CogVideoX I2V 是否有灯珠
2. 如果没有灯珠，将 VideoPainter 代码移植到新版 diffusers
3. 如果仍有，则是 PyTorch 2.10.0 的问题，需要降级


● 新 notebook 写好了：VideoPainter/VideoPainter_Colab_Official.ipynb

  和之前的 notebook 有什么不同

  ┌──────────────┬──────────────────────────┬────────────────────────────────┐
  │     项目      │     之前的 notebook      │              这次              │
  ├──────────────┼──────────────────────────┼────────────────────────────────┤
  │ PyTorch      │ 2.10.0（Colab 默认）     │ 2.4.0（官方要求）                |
  ├──────────────┼──────────────────────────┼────────────────────────────────┤
  │ diffusers    │ sys.path hack            │ pip install -e（官方方式）      |
  ├──────────────┼──────────────────────────┼────────────────────────────────┤
  │ transformers │ Colab 默认版本           │ 4.42.2（官方要求）               │
  ├──────────────┼──────────────────────────┼────────────────────────────────┤
  │ numpy        │ Colab 默认               │ 1.26.0（官方要求）               │
  ├──────────────┼──────────────────────────┼────────────────────────────────┤
  │ 推理逻辑     │ 自写 run_demo.py         │ 复刻 infer/inpaint.py 的逻辑     │
  ├──────────────┼──────────────────────────┼────────────────────────────────┤
  │ GPU 加载     │ enable_model_cpu_offload │ pipe.to("cuda")（官方方式）      │
  ├──────────────┼──────────────────────────┼────────────────────────────────┤
  │ VAE 优化     │ slicing + tiling         │ 不开（官方非 long_video 不开）   │
  └──────────────┴──────────────────────────┴────────────────────────────────┘
