# GT Inspection

本目录用于保存 SAM3 PVTT 复现过程中抽帧结果的 GT 核对材料，便于人工检查数据集标注是否存在问题。

## 目录结构

```
GT_Inspection/
├── README.md                  # 本说明
├── export_necklace_frames.py  # necklace 导出脚本
└── necklace/                  # 0034-necklace1 的 8 帧核对结果
    ├── summary.json
    ├── frame_XX.jpg
    ├── gt_mask_XX.jpg
    ├── pred_mask_XX.png
    ├── overlay_gt_only_XX.jpg
    ├── overlay_pred_only_XX.jpg
    ├── overlay_gt_pred_XX.jpg
    └── compare_XX.jpg
```

---

## Necklace（0034-necklace1）

对应 `sam3_qualitative.pdf` 中 panel **(c)** 的低分样本，视频为 **`0034-necklace1`**（mean IoU = 0.058，8 帧均匀采样）。

- **视频路径**：`/data/datasets/pvtt_new/pvtt-evaluation/datasets_new/source_videos_100/0034-necklace1.mp4`
- **GT mask 目录**：`/data/datasets/pvtt_new/pvtt-evaluation/datasets_new/masks/0034-necklace1/`（共 384 帧）
- **复现来源**：`/data/liuluyan/sam3/sam3_pvtt_full/`
- **panel (c) 使用帧**：`frame_01.jpg`（`compare_01.jpg`）

### 叠加颜色约定

| 颜色 | 含义 |
|------|------|
| 绿色 | 数据集 GT mask |
| 蓝色 | SAM3 预测 mask |

### 每帧文件说明（XX = 00–07）

| 文件 | 内容 |
|------|------|
| `frame_XX.jpg` | 从视频按复现脚本均匀采样的原帧 |
| `gt_mask_XX.jpg` | 数据集原始 GT mask（未叠加） |
| `pred_mask_XX.png` | SAM3 预测二值 mask |
| `overlay_gt_only_XX.jpg` | 原帧 + 仅 GT 叠加（绿色） |
| `overlay_pred_only_XX.jpg` | 原帧 + 仅预测叠加（蓝色） |
| `overlay_gt_pred_XX.jpg` | 原帧 + GT + 预测叠加（与复现 overlay 一致） |
| `compare_XX.jpg` | 四宫格对比：原帧 \| GT \| 预测 \| GT+预测 |

**建议优先查看**：`compare_XX.jpg` 或 `overlay_gt_only_XX.jpg`，用于判断 GT 是否与画面中项链对齐。

### 各帧 IoU 与 GT 对应关系

| 帧 | 时间 (s) | GT 帧索引 | GT 文件 | IoU |
|----|----------|-----------|---------|-----|
| 00 | 0.21 | 24 | `00025.jpg` | 0.165 |
| **01** | **0.63** | **72** | **`00073.jpg`** | **0.061** ← panel (c) |
| 02 | 1.04 | 120 | `00121.jpg` | 0.040 |
| 03 | 1.46 | 168 | `00169.jpg` | 0.040 |
| 04 | 1.88 | 215 | `00216.jpg` | 0.048 |
| 05 | 2.29 | 263 | `00264.jpg` | 0.045 |
| 06 | 2.71 | 311 | `00312.jpg` | 0.033 |
| 07 | 3.13 | 359 | `00360.jpg` | 0.036 |

- **mean IoU**：0.058
- **median IoU**：0.042

### 初步观察（frame_01 / panel c）

在 `compare_01.jpg` 中：

- **绿色 GT** 呈现粗线条，与画面中项链链/十字架的位置明显不对齐。
- **蓝色 SAM3 预测** 反而更贴近实际项链轮廓。

8 帧的 GT 均呈现类似的粗线、偏移特征，更像是数据集标注问题，而非单帧偶然错误。

### 重新导出

```bash
source /data/liuluyan/miniconda3/etc/profile.d/conda.sh
conda activate sam3
python /data/liuluyan/sam3/report_figures/GT_Inspection/export_necklace_frames.py
```

脚本会从 `sam3_pvtt_full` 的 frames / pred_masks / overlays 及数据集 GT 重新生成 `necklace/` 下全部文件。
