## 3. 视频目标分割

### 3.1 SAM 2 (Segment Anything Model 2)

| 字段 | 信息 |
|------|------|
| **论文** | SAM 2: Segment Anything in Images and Videos |
| **arXiv** | [2408.00714](https://arxiv.org/abs/2408.00714) |
| **机构** | Meta AI (FAIR) |
| **代码** | [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2) |

#### 完整架构

**图像编码器**:
- MAE预训练的Hiera分层架构
- 特征金字塔网络融合Stage 3和Stage 4的stride-16/stride-32特征
- 窗口化绝对位置嵌入
- 变体: Tiny (T), Small (S), Base+ (B+), Large (L)
- 空间分辨率: 512, 768, 1024像素

**记忆注意力 (默认L=4个Transformer块)**:
- 每块模式: 自注意力 → 对记忆+对象指针的交叉注意力 → MLP
- 2D空间旋转位置编码 (RoPE)

**提示编码器与掩码解码器**:
- 支持点击、边界框、掩码作为提示
- "双向"Transformer块更新提示和帧嵌入
- **遮挡预测头**: 额外token + MLP产生可见性概率 — 处理目标临时消失

**记忆编码器**:
- 复用Hiera图像嵌入，无需独立编码器
- 通过卷积下采样输出掩码
- 投影至64维记忆特征

**记忆库**:
- FIFO队列: 最近N=6帧的记忆
- 独立FIFO队列: 最多M个提示帧
- 存储空间特征图 + 轻量对象指针向量 (256维，分为4×64维token)

#### 视频传播流程

1. 图像编码器处理当前帧（整个交互会话只需一次）
2. 提供无条件特征嵌入作为token
3. 记忆注意力将帧嵌入条件化于记忆库
4. 解码器接受条件化帧嵌入 + 可选提示 → 输出分割掩码
5. 记忆编码器转换预测 + 图像嵌入用于后续帧
6. 掩码下采样、融合、存入FIFO队列
7. 对象指针token存储用于后续帧的交叉注意力

#### SA-V数据集

- **规模**: 50.9K视频, 642.6K masklets, 35.5M掩码 (比任何先前VOS数据集大53倍)
- **三阶段数据引擎**:
  - Phase 1 (SAM逐帧): 37.8秒/帧, 16K masklets
  - Phase 2 (SAM + SAM2掩码): 7.4秒/帧 (5.1×加速)
  - Phase 3 (完整SAM2): 4.5秒/帧 (8.4×加速)

#### 基准性能

| 基准 | J&F |
|------|-----|
| DAVIS 2017 val | 90.9-91.6 |
| YouTube-VOS 2019 | 88.4-89.1 |
| MOSE val | 75.8-77.2 |
| 推理速度 | Hiera-B+ @ 1024: 43.8 FPS; Hiera-L: 30.2 FPS |

**与PVTT的关联**: 主要的视频掩码传播工具。Grounded-SAM2检测产品后，SAM2的流式记忆架构将分割掩码传播到所有后续帧。遮挡预测头对电商视频中产品被部分遮挡的场景至关重要。

### 3.2 Grounded-SAM2

| 字段 | 信息 |
|------|------|
| **论文** | Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks |
| **arXiv** | [2401.14159](https://arxiv.org/abs/2401.14159) |
| **代码** | [github.com/IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |

#### 工作流程

1. **文本提示** (如"handbag", "shoe", "product") → **Grounding DINO** 检测边界框
2. SAM 2 从检测框生成分割掩码
3. SAM 2 视频传播跨所有帧跟踪掩码

#### Grounding DINO 1.5 ([2405.10300](https://arxiv.org/abs/2405.10300))

- **Pro模型**: COCO 54.3 AP, LVIS-minival 零样本55.7 AP
- **Edge模型**: TensorRT下75.2 FPS, LVIS-minival零样本36.2 AP
- 深度早期融合架构，20M+带标注图像训练

**与PVTT的关联**: 管线第一步。文本提示 → Grounding DINO检测产品边界框 → SAM2生成并传播分割掩码。55.7 AP LVIS零样本意味着可处理多种电商产品类别。

### 3.3 其他视频目标分割方法

**Cutie** (CVPR 2024 Highlight) - [GitHub](https://github.com/hkchengrex/Cutie):
- 对象记忆 + 对象Transformer实现双向信息交互
- 在严重遮挡场景下优于XMem，可作为后备方案

**XMem** ([2207.07042](https://arxiv.org/abs/2207.07042), ECCV 2022):
- Atkinson-Shiffrin记忆模型：感觉记忆、工作记忆和长期记忆
- 三个独立记忆库，擅长长视频处理

**DEVA** - "Tracking Anything with Decoupled Video Segmentation":
- 解耦为图像级分割 + 时序传播
- 允许插入任意通用图像分割模型

---

## 4. 视频修复与背景恢复

### 4.1 VideoPainter

| 字段 | 信息 |
|------|------|
| **论文** | VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control |
| **arXiv** | [2503.05639](https://arxiv.org/abs/2503.05639) |
| **会议** | SIGGRAPH 2025 |
| **机构** | TencentARC |
| **代码** | [github.com/TencentARC/VideoPainter](https://github.com/TencentARC/VideoPainter) |

#### 双分支架构

- **上下文编码器**: 仅使用预训练DiT的前2层（骨干参数的6%）。处理拼接输入：噪声潜变量 + 掩码视频潜变量 + 下采样掩码
- **Token选择性集成**: 仅纯背景token被添加回去；前景token被排除。基于分割掩码的预过滤防止前景-背景歧义
- **冻结DiT骨干** + 可训练上下文编码器

#### 目标区域ID重采样（任意长度视频）

- 训练阶段: 冻结DiT + 上下文编码器; 可训练ID-Resample Adapters (LoRA)
- 推理阶段: 前一片段的修复区域token拼接到当前KV对，维持长视频ID一致性

#### VPData数据集 (390K+片段, >866.7小时)

- **收集**: ~450K视频来自Videvo和Pexels
- **标注管线 (5步)**:
  1. 收集: 获取原始视频
  2. 标注: 级联式 — Recognize Anything Model → Grounding DINO → SAM2
  3. 过滤: 帧间掩码面积变化 Δ < 20%，帧覆盖率30-70%
  4. 分割: PySceneDetect场景转换，10秒间隔，丢弃<6秒片段
  5. 选择: LAION美学评分，RAFT光流（运动强度），SD安全检查

#### 性能

| 基准 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| VPBench-S | 23.32 | 0.89 | 6.85e-2 |
| VPBench-L | 22.19 | 0.85 | — |
| DAVIS | 25.27 | 0.94 | — |

- 用户研究 (30人): 背景保持74.2%, 文本对齐82.5%, 视频质量87.4%
- **即插即用**: 可与任何预训练DiT（T2V和I2V）配合使用

**与PVTT的关联**: 主要的背景恢复工具。给定Grounded-SAM2的产品掩码，VideoPainter修复被遮挡区域生成时序一致的背景。其即插即用设计可与Wan2.1/2.2骨干配合。VPData构建管线（RAM + Grounding DINO + SAM2 + PySceneDetect + 美学评分）与PVTT计划的工具链几乎完全一致。

### 4.2 ProPainter

| 字段 | 信息 |
|------|------|
| **论文** | ProPainter: Improving Propagation and Transformer for Video Inpainting |
| **arXiv** | [2309.03897](https://arxiv.org/abs/2309.03897) |
| **会议** | ICCV 2023 |
| **代码** | [github.com/sczhou/ProPainter](https://github.com/sczhou/ProPainter) |

#### 三组件架构

1. **递归光流补全**: 高效递归网络补全损坏的光流场
2. **双域传播**: 结合图像扭曲和特征扭曲利用全局对应关系
3. **掩码引导稀疏视频Transformer**: 基于掩码引导丢弃不必要窗口实现高效

**性能**: 比先前方法提升 **1.46 dB PSNR**

**与PVTT的关联**: 基于光流的替代/补充方案。适合短片段或需要确定性结果时使用。非生成式方法，对干净背景恢复可能更优。

### 4.3 其他视频修复方法

**DiffuEraser** ([2501.10018](https://arxiv.org/abs/2501.10018), 2025年1月):
- 基于Stable Diffusion + 辅助BrushNet分支
- 使用ProPainter输出作为先验初始化

**EraserDiT** ([2506.12853](https://arxiv.org/abs/2506.12853), 2025年6月):
- 基于DiT的视频修复 + **环形位置偏移**策略
- 自动检测目标、交互式移除

---
