# INFER_NO_MASK 重跑结果分析

针对原复现 `sam3_pvtt_full` 中 23 帧 `INFER_NO_MASK` 失败，按建议策略重跑；**其余 761 帧从原目录原样复制**。

## 改动的 Prompt（仅 2 处）

| overlays 文件夹 | 原 prompt | 重跑 prompt | 说明 |
|----------------|-----------|-------------|------|
| `0013-handbag3_scene03/` | `handbag` | **`backpack`** | 8 帧全部改用；目标为双肩背包 |
| `0013-handbag3_scene01/`<br>`0013-handbag3_scene02/` | `handbag` | **`sling bag`** | 背戴斜挎包场景；scene02 8/8 成功，scene01 部分帧成功 |

**未改 prompt 的说明：**
- `0038-necklace5/` 仍为 `necklace`，只把阈值从 0.1 降到 0.05
- 其余 94 个视频文件夹 prompt 未动

---

## 先说清楚：overlay 里那么多文件夹是什么？

`no_mask_retry/overlays/` 下共有 **98 个子文件夹**，对应 PVTT 完整评测的 **98 个视频**，命名规则为 `{视频名}/`，例如：

```
overlays/
├── 0013-handbag3_scene01/    ← 本次重跑涉及（4 个之一）
├── 0013-handbag3_scene02/    ← 本次重跑涉及
├── 0013-handbag3_scene03/    ← 本次重跑涉及
├── 0014-handbag4/            ← 未动，从 sam3_pvtt_full 原样复制
├── 0015-handbag5_scene01/    ← 未动，原样复制
├── …（共 98 个）
├── 0038-necklace5/           ← 本次重跑涉及（4 个之一）
└── 0050-clothing2/           ← 未动，原样复制
```

**本文所有指标只涉及以下 4 个文件夹**（原 23 帧失败的来源），其余 94 个文件夹数据与 `sam3_pvtt_full/overlays/{同名}/` 完全一致，**未做任何重跑**：

| 文件夹（overlays 下） | 商品 | 原失败帧 | 重跑后 |
|----------------------|------|----------|--------|
| `0013-handbag3_scene01/` | 棕色斜挎包（行走 scene 1） | 6/8 | 5/8 成功，3 仍失败 |
| `0013-handbag3_scene02/` | 棕色斜挎包（行走 scene 2） | 8/8 | 8/8 成功 |
| `0013-handbag3_scene03/` | 黑色双肩背包 + 摩托车 scene | 8/8 | 8/8 成功 |
| `0038-necklace5/` | 项链 | 1/8（仅 frame_04） | 8/8 成功 |

> **关于 "scene01/02/03" 的说明**：这不是全局第 1/2/3 场景，而是**同一商品 `0013-handbag3` 的 3 段不同拍摄视频**，数据集命名为 `0013-handbag3_scene01`、`0013-handbag3_scene02`、`0013-handbag3_scene03`。下文一律用**完整文件夹名**，避免歧义。

每个文件夹内有 8 张图：`frame_00.jpg` … `frame_07.jpg`（均匀采样的 8 帧 overlay）。

---

## 目录结构

```
no_mask_retry/
├── README.md
├── rerun_failed.py
├── retry_results.json       # 23 帧逐次尝试日志
├── frames/                  # 软链 → ../sam3_pvtt_full/frames
├── pred_masks/{98 视频}/    # 94 视频原样复制 + 上表 4 视频部分更新
├── overlays/{98 视频}/      # 同上
└── meta/{98 视频}.json
```

---

## 整体 IoU（98 视频 × 8 帧 = 784 帧）

统计来源：`meta/*.json` 中各帧 IoU 汇总。

| 指标 | 原复现 `sam3_pvtt_full` | 重跑后 `no_mask_retry` | 变化 |
|------|-------------------------|------------------------|------|
| 有效帧数 | 761 / 784 | **781 / 784** | +20 帧 |
| 无 mask 失败帧 | 23（2.93%） | **3（0.38%）** | −20 |
| **Micro mean IoU**（全部有效帧 IoU 算术平均） | **0.6184** | **0.6192** | +0.0008 |
| **Micro median IoU**（全部有效帧 IoU 中位数） | **0.7429** | **0.7486** | +0.0057 |
| **Macro mean IoU**（98 视频各自 mean IoU 再平均） | **0.6205** | **0.6203** | −0.0002 |

Micro mean 略升（多了 20 帧成功推理）；Macro mean 几乎不变，因新增帧来自原本 IoU=0 的视频，对 per-video 均值拉动有限。

### 4 个涉及文件夹的 per-video IoU 对比

| overlays 文件夹 | 原 n_ok | 原 mean IoU | 重跑 n_ok | 重跑 mean IoU |
|-----------------|---------|-------------|-----------|---------------|
| `0013-handbag3_scene01/` | 2/8 | 0.9267 | 5/8 | 0.9184 |
| `0013-handbag3_scene02/` | 0/8 | 0.0000 | 8/8 | **0.9374** |
| `0013-handbag3_scene03/` | 0/8 | 0.0000 | 8/8 | **0.3039** |
| `0038-necklace5/` | 7/8 | 0.3919 | 8/8 | 0.3854 |

> scene03 重跑后 mean IoU 仅 0.30：mask 已全部恢复，但 `"backpack"` 与 GT（仍标注为 handbag 区域）对齐较差，多帧 IoU < 0.15。

---

## 重跑策略（仅上述 4 个文件夹内的失败帧）

| 实验 | overlays 文件夹 | 原失败帧 | 尝试顺序 |
|------|-----------------|----------|----------|
| 1 | `0013-handbag3_scene03/` | 8 帧全失败 | `backpack@0.1` → `backpack@0.05` |
| 2 | `0013-handbag3_scene02/` | 8 帧全失败 | `messenger bag@0.1` → `sling bag@0.1` → `handbag@0.05` → `handbag@0.01` |
| 2' | `0013-handbag3_scene01/` | 6 帧失败 | 同实验 2 |
| 3 | `0038-necklace5/` | 仅 `frame_04.jpg` | `necklace@0.1` → `necklace@0.05` → `necklace@0.01` |

每帧取**第一个**产生 mask 的配置作为最终结果，写入对应文件夹的 `pred_masks/` 与 `overlays/`。

---

## 实验 1：`overlays/0013-handbag3_scene03/` → prompt `backpack`（8/8 恢复）

**结论：原 prompt `"handbag"` 对该文件夹内黑色双肩背包完全无效；改 `"backpack"` 后 8 帧均有 mask。**

| 文件 | 生效配置 | IoU |
|------|----------|-----|
| `frame_00.jpg` | backpack@0.1 | 0.793 |
| `frame_01.jpg` | backpack@0.1 | 0.464 |
| `frame_02.jpg` | backpack@0.1 | 0.449 |
| `frame_03.jpg` | backpack@0.1 | 0.063 |
| `frame_04.jpg` | backpack@0.1 | 0.373 |
| `frame_05.jpg` | backpack@0.1 | 0.037 |
| `frame_06.jpg` | backpack@0.1 | 0.106 |
| `frame_07.jpg` | backpack@0.1 | 0.146 |

该文件夹 mean IoU = **0.3039**（mask 全恢复，但部分帧定位/GT 对齐差）。

---

## 实验 2：`overlays/0013-handbag3_scene02/` → prompt `sling bag`（8/8 恢复）

**结论：原 `"handbag"` 与背戴斜挎包语义不匹配；`sling bag@0.1` 对 7/8 帧有效。**

| 生效配置 | 帧数 | IoU 范围 |
|----------|------|----------|
| `sling bag@0.1` | 7 | 0.853 – 0.972 |
| `handbag@0.01` | 1（`frame_03.jpg`） | 0.940 |

- `messenger bag@0.1`：该文件夹 **0/8 成功**
- 该文件夹重跑后 mean IoU = **0.9374**

---

## 实验 2'：`overlays/0013-handbag3_scene01/`（3/6 失败帧恢复，3 帧仍失败）

与 `scene02` 同类斜挎包；原复现该文件夹 8 帧中 `frame_00.jpg`、`frame_04.jpg` 已成功（IoU 0.89–0.96），未重跑。

| 文件 | 结果 | 生效配置 | IoU |
|------|------|----------|-----|
| `frame_01.jpg` | 恢复 | handbag@0.01 | 0.922 |
| `frame_02.jpg` | **仍失败** | 全部无效 | — |
| `frame_03.jpg` | **仍失败** | 全部无效 | — |
| `frame_05.jpg` | **仍失败** | 全部无效 | — |
| `frame_06.jpg` | 恢复 | sling bag@0.1 | 0.911 |
| `frame_07.jpg` | 恢复 | sling bag@0.1 | 0.905 |

该文件夹重跑后：**5/8 成功，mean IoU = 0.9184**；仍失败 3 帧见 `frames/0013-handbag3_scene01/frame_02|03|05.jpg`（运动模糊 + 极端背视角，prompt/阈值已耗尽）。

---

## 实验 3：`overlays/0038-necklace5/frame_04.jpg` → 降阈值（1/1 恢复）

| 配置 | 结果 |
|------|------|
| necklace@0.1 | 失败 |
| necklace@0.05 | **成功**，IoU = 0.340 |

该文件夹 8 帧全部成功，mean IoU = **0.3854**（与原 0.3919 接近）。

---

## 原失败原因验证（仅针对上述 4 文件夹）

| 原假设 | 对应文件夹 | 验证 |
|--------|------------|------|
| prompt 与双肩背包不符 | `0013-handbag3_scene03/` | **确认** — `backpack` 8/8 恢复 |
| prompt 与背戴斜挎包不符 | `0013-handbag3_scene02/` | **确认** — `sling bag` 7/8 |
| 置信度阈值边界 | `0013-handbag3_scene01/`、`0038-necklace5/` | **部分确认** — necklace 降阈值即可；scene01 仍剩 3 帧 |
| 目标被遮挡 | 上述 handbag 文件夹 | **否** |
| `messenger bag` prompt | scene01 + scene02 | **无效**（0/22） |

## 建议

1. 背戴斜挎包 → prompt 用 `"sling bag"`（文件夹 `0013-handbag3_scene01/`、`scene02/`）
2. 双肩背包 → prompt 用 `"backpack"`（文件夹 `0013-handbag3_scene03/`）
3. 置信度阈值 0.1 偏严，可考虑 0.05
4. 仍失败 3 帧路径：`frames/0013-handbag3_scene01/frame_02.jpg`、`frame_03.jpg`、`frame_05.jpg`

## 复现

```bash
source /data/liuluyan/miniconda3/etc/profile.d/conda.sh
conda activate sam3
CUDA_VISIBLE_DEVICES=4 python /data/liuluyan/sam3/sam3_pvtt_full/no_mask_retry/rerun_failed.py
```

逐帧尝试细节见 `retry_results.json`。
