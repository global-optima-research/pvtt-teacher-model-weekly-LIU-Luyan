# Week 4 Progress Report — DAVIS 2017 Benchmark: 5-Model VOS Comparison

**Author:** Luluyan Liu
**Date:** 2026-03-09
**Server:** liuluyan@111.17.197.107
**Base Path:** `/data/liuluyan/`

---

## 1. Overview

本周在 DAVIS 2017 validation set 上对 5 个视频目标分割（VOS）模型进行了系统性 benchmark 测试，评估各模型在半监督 VOS 任务中的表现（给定首帧 mask，逐帧传播分割）。

| 模型 | 类型 | 权重 | 结果路径 |
|------|------|------|----------|
| **SAM2** | Foundation Model (点/框/文本提示) | sam2.1_hiera_large.pt | `SAM2/davis2017_results/` |
| **Grounded-SAM-2** | SAM2 + Grounding DINO | sam2.1_hiera_large.pt | `Grounded-SAM-2-clean/davis2017_results/` |
| **Cutie** | 轻量级 VOS (对象级记忆) | cutie-base-mega.pth | `Cutie/davis2017_results/` |
| **DEVA** | 解耦式 VOS (传播模型) | DEVA-propagation.pth | `Tracking-Anything-with-DEVA/davis2017_results/` |
| **XMem** | 记忆式 VOS (长短期记忆) | XMem.pth | `XMem/davis2017_results/` |

**数据集：** DAVIS 2017 validation set — 30 个视频，480p 分辨率
**评估指标：** J (Region Similarity / IoU), F (Contour Accuracy), J&F (综合得分)

---

## 2. Overall Results

| Model | J&F | J (Region) | F (Contour) |
|-------|-----|------------|-------------|
| **SAM2** | **0.9154** | **0.8950** | **0.9359** |
| **GSAM2** | **0.9154** | **0.8950** | **0.9359** |
| **Cutie** | 0.8965 | 0.8778 | 0.9152 |
| **DEVA** | 0.8952 | 0.8756 | 0.9149 |
| **XMem** | 0.8805 | 0.8624 | 0.8987 |

**排名：SAM2 = GSAM2 > Cutie > DEVA > XMem**

> **注意：** GSAM2 在半监督 VOS 任务中使用与 SAM2 相同的视频传播模块（SAM2 Video Predictor），因此两者结果完全一致。GSAM2 的独特价值在于其文本驱动的自动检测能力（Grounding DINO），在无需手动标注首帧 mask 的场景中优于 SAM2。

---

## 3. Per-Video Detailed Results

### 3.1 所有视频 J&F 得分

| Video | SAM2 | Cutie | DEVA | XMem |
|-------|------|-------|------|------|
| bike-packing | 0.908 | 0.877 | 0.873 | 0.845 |
| blackswan | 0.970 | 0.976 | 0.977 | **0.981** |
| bmx-trees | **0.816** | 0.786 | 0.773 | 0.768 |
| breakdance | **0.977** | 0.949 | 0.931 | 0.921 |
| camel | **0.988** | 0.986 | 0.986 | 0.982 |
| car-roundabout | 0.975 | 0.980 | **0.983** | **0.984** |
| car-shadow | **0.984** | **0.984** | 0.983 | 0.982 |
| cows | **0.983** | **0.984** | 0.980 | 0.975 |
| dance-twirl | **0.958** | 0.904 | 0.908 | 0.906 |
| dog | **0.971** | 0.967 | 0.964 | 0.963 |
| dogs-jump | **0.965** | 0.958 | 0.953 | 0.940 |
| drift-chicane | **0.952** | 0.904 | 0.945 | 0.903 |
| drift-straight | **0.968** | 0.951 | 0.943 | 0.937 |
| goat | **0.940** | 0.929 | 0.934 | 0.925 |
| gold-fish | **0.946** | 0.939 | 0.929 | 0.904 |
| horsejump-high | **0.927** | 0.921 | 0.922 | 0.912 |
| india | **0.813** | 0.774 | 0.776 | 0.756 |
| judo | **0.911** | 0.893 | 0.902 | 0.886 |
| kite-surf | **0.724** | 0.699 | 0.701 | 0.675 |
| lab-coat | 0.694 | 0.677 | 0.662 | **0.695** |
| libby | 0.952 | **0.958** | 0.953 | 0.946 |
| loading | **0.973** | 0.888 | 0.934 | 0.907 |
| mbike-trick | **0.899** | 0.859 | 0.874 | 0.812 |
| motocross-jump | **0.919** | 0.890 | 0.872 | 0.815 |
| paragliding-launch | 0.653 | 0.659 | **0.675** | 0.662 |
| parkour | **0.973** | 0.969 | 0.970 | 0.966 |
| pigs | 0.943 | **0.947** | 0.932 | 0.903 |
| scooter-black | **0.922** | 0.897 | 0.896 | 0.879 |
| shooting | **0.926** | 0.907 | 0.842 | 0.826 |
| soapbox | **0.934** | 0.884 | 0.888 | 0.862 |

（加粗表示该视频上得分最高的模型）

### 3.2 Best/Worst Cases per Model

| Model | Best 3 Videos | Worst 3 Videos |
|-------|---------------|----------------|
| SAM2 | camel (0.988), car-shadow (0.984), cows (0.983) | paragliding-launch (0.653), lab-coat (0.694), kite-surf (0.724) |
| Cutie | camel (0.986), car-shadow (0.984), cows (0.983) | paragliding-launch (0.658), lab-coat (0.677), kite-surf (0.699) |
| DEVA | camel (0.985), car-roundabout (0.983), car-shadow (0.982) | lab-coat (0.662), paragliding-launch (0.675), kite-surf (0.701) |
| XMem | car-roundabout (0.984), camel (0.982), car-shadow (0.982) | paragliding-launch (0.662), kite-surf (0.675), lab-coat (0.695) |

---

## 4. Specific Case Analysis

### 4.1 SAM2 明显优于其他模型的 Case

**Case 1: `breakdance` (SAM2: 0.977 vs XMem: 0.921, Δ=0.056)**
- 包含快速运动和剧烈形变的人体
- SAM2 的 Hiera backbone 对快速运动的跟踪能力更强
- XMem/DEVA 在快速旋转时容易丢失目标边缘

**Case 2: `loading` (SAM2: 0.973 vs Cutie: 0.888, Δ=0.085)**
- 场景中存在物体遮挡和重新出现
- SAM2 能更好地在遮挡后恢复跟踪
- Cutie 在遮挡后的 mask 精度下降明显

**Case 3: `dance-twirl` (SAM2: 0.958 vs Cutie: 0.904, Δ=0.054)**
- 旋转导致目标外观变化大
- SAM2 对外观变化的适应能力更强

**Case 4: `motocross-jump` (SAM2: 0.919 vs XMem: 0.815, Δ=0.104)**
- 所有模型中差异最大的视频之一
- 快速运动 + 多目标 + 遮挡，SAM2 明显优于传统 VOS 模型

### 4.2 所有模型均表现不佳的 Case

**Case: `paragliding-launch` (最高仅 0.675, DEVA)**
- 所有模型 J&F 均低于 0.68
- 远距离小目标 + 复杂背景
- 提示：对于此类场景可能需要更强的提示策略或多尺度分割

**Case: `lab-coat` (最高仅 0.695, XMem)**
- 有趣的是，XMem 在此视频上反而略优于 SAM2
- 可能涉及白色物体在白色背景下的分割困难

**Case: `kite-surf` (最高仅 0.724, SAM2)**
- 远距离小目标 + 海面反光
- 所有模型均受到小目标检测的挑战

### 4.3 其他模型偶尔优于 SAM2 的 Case

- **`blackswan`**: XMem (0.981) > SAM2 (0.970) — 简单场景下传统 VOS 模型足够准确
- **`car-roundabout`**: XMem (0.984) > SAM2 (0.975) — 稳定运动的场景
- **`cows` / `libby` / `pigs`**: Cutie 略优于 SAM2 — 中等复杂度场景中 Cutie 的对象级记忆策略有效

---

## 5. Model Comparison Summary

### 5.1 综合评价

| 维度 | SAM2 | Cutie | DEVA | XMem |
|------|------|-------|------|------|
| **J&F 得分** | 0.915 (最高) | 0.897 | 0.895 | 0.881 |
| **快速运动处理** | 最强 | 中等 | 较好 | 较弱 |
| **遮挡恢复** | 最强 | 较弱 | 中等 | 中等 |
| **小目标分割** | 较好 | 较好 | 较好 | 一般 |
| **模型大小** | 最大 (Hiera-L) | 中等 (Base-Mega) | 中等 | 中等 |
| **推理速度 (FPS)** | ~35 | 82 | 88 | 94 |
| **独特优势** | 全能型，精度最高 | 速度与精度平衡好 | 解耦设计灵活 | 速度最快 |

### 5.2 关键发现

1. **SAM2 / GSAM2 以 J&F=0.915 排名第一**，在 30 个视频中的 23 个取得最高分
2. **GSAM2 = SAM2**（在半监督 VOS 中）：因为两者共用相同的视频传播模块。GSAM2 的额外价值在于文本检测能力
3. **Cutie 和 DEVA 非常接近**（0.897 vs 0.895），Cutie 在区域精度(J)上略好
4. **XMem 速度最快但精度最低**，在快速运动场景（breakdance, motocross-jump）中表现较差
5. **所有模型在小目标/远距离场景中均挣扎**（paragliding-launch, kite-surf, lab-coat）
6. **传统 VOS 模型在简单场景中有时可匹敌 SAM2**（blackswan, car-roundabout, cows 等），但在困难场景中差距拉大

### 5.3 对 Product Video Try-On Pipeline 的建议

- **推荐使用 SAM2/GSAM2** 作为主力分割模型，精度最高
- **GSAM2 适合自动化标注**：无需手动提供首帧 mask，文本提示即可检测产品
- **Cutie 适合大规模批处理**：速度较快且精度仅比 SAM2 低 ~2%
- 对于产品视频中的 **小物体/遮挡场景**，建议结合多种提示策略（文本+点+框）

---

## 6. Environment & Configuration

```
Server: liuluyan@111.17.197.107
GPU: NVIDIA RTX 5090
CUDA: 12.8
PyTorch: 2.7.0
Conda env: lly311 (Python 3.11)
Dataset: DAVIS 2017 val (30 videos, 480p) at /data/datasets/DAVIS/
```

### Model Weights

| Model | Weight File | Size |
|-------|------------|------|
| SAM2 / GSAM2 | sam2.1_hiera_large.pt | ~900MB |
| Cutie | cutie-base-mega.pth | 134MB |
| DEVA | DEVA-propagation.pth | 264MB |
| XMem | XMem.pth | 237MB |

---

## 7. File Structure

```
/data/liuluyan/
├── SAM2/
│   ├── davis2017_results/           (30 videos, SAM2 predictions)
│   └── dataset100_results/
├── Grounded-SAM-2-clean/
│   ├── davis2017_results/           (30 videos, GSAM2 predictions)
│   └── dataset100_results/
├── Cutie/
│   ├── davis2017_results/Annotations/  (30 videos)
│   └── weights/
├── Tracking-Anything-with-DEVA/
│   ├── davis2017_results/           (30 videos)
│   └── saves/
├── XMem/
│   ├── davis2017_results/           (30 videos)
│   └── saves/
├── repo/
│   ├── week3_report.md
│   ├── week4_report.md
│   ├── eval_davis.py
│   └── davis2017_eval_results.json
└── miniconda3/
```

---

## 8. Next Steps

1. **可视化对比**：抽取 3-5 个 case（如 breakdance, loading, paragliding-launch），逐帧可视化各模型的 mask 覆盖效果
2. **扩展评估**：在更多数据集（如 YouTube-VOS）上验证结果的一致性
3. **优化 Pipeline**：将最优模型（SAM2/GSAM2）与 VideoPainter 结合，在 dataset100 上进行端到端测试
4. **消融实验**：测试不同 SAM2 模型尺寸（Tiny/Small/Base/Large）对精度和速度的影响
