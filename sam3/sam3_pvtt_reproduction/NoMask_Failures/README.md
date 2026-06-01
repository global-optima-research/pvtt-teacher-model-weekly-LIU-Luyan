# INFER_NO_MASK 失败帧分析

SAM3 PVTT 完整复现（98 视频 × 8 帧 = 784 次推理）中，共 **23 帧** 未返回任何 mask（`failed.txt` 全部为 `INFER_NO_MASK`），失败率 **2.93%**。

- 置信度阈值：`confidence_threshold = 0.1`（低于此值的检测会被丢弃，视为无 mask）
- 本目录保存这 23 帧的**原图**及 **GT 叠加图**（绿色），便于人工核对

## 文件命名

| 文件 | 说明 |
|------|------|
| `{video}_frame_{XX}.jpg` | 采样原帧 |
| `{video}_frame_{XX}_gt_overlay.jpg` | 原帧 + 数据集 GT（绿色） |
| `summary.json` | 机器可读清单（prompt、GT 路径等） |

---

## 失败分布

| 视频 | 类别 | prompt | 失败帧数 | 成功帧数 |
|------|------|--------|----------|----------|
| `0013-handbag3_scene01` | handbag | `handbag` | 6 | 2（frame_00, frame_04） |
| `0013-handbag3_scene02` | handbag | `handbag` | 8 | 0 |
| `0013-handbag3_scene03` | handbag | `handbag` | 8 | 0 |
| `0038-necklace5` | necklace | `necklace` | 1（frame_04） | 7 |

**结论：23 帧失败全部来自 4 个视频，其中 22 帧集中在同一商品的 3 个 handbag 场景。**

---

## 原因分析

### 1. Prompt 语义不匹配（主因，scene03 全部 8 帧）

`0013-handbag3_scene03` 中目标物是**黑色双肩背包（backpack）**，而非手提包。人物骑/靠近摩托车，背包在背上清晰可见。

- prompt 为 `"handbag"`，SAM3 在该语义下**全程无输出**
- 同系列 scene01 中，背上的**棕色皮质斜挎/胸包**有时能被 `"handbag"` 检出（2/8 成功），说明 scene03 的**背包形态与 prompt 差距更大**
- **不是遮挡**：背包在画面中完整可见

### 2. 视角 + 佩戴方式导致类别歧义（scene02 全部 8 帧）

`0013-handbag3_scene02` 目标为**棕色皮质斜挎包**，佩戴在**人物背部**，全程背对镜头行走。

- 物体**未被遮挡**，GT 标注区域与包的位置一致
- SAM3 对 `"handbag"` **8 帧全部无输出**，而同形态的包在 scene01 偶发成功 → 更可能是**背对镜头 + 斜挎佩戴**使物体视觉类别接近 messenger bag / sling bag，与 `"handbag"` 训练先验不一致
- **不是 prompt 完全错误**（scene01 有成功案例），而是**呈现方式使检测置信度持续低于 0.1**

### 3. 置信度边界波动（scene01 中 6 帧）

`0013-handbag3_scene01` 与 scene02 为**同类棕色斜挎包、同类行走场景**，但 8 帧中 **frame_00、frame_04 成功**（IoU 0.89–0.96），其余 6 帧失败。

- 成功帧与失败帧在视觉上非常接近（均为背对镜头、包在背上）
- 差异主要来自**步态/角度细微变化、运动模糊、包在背上占比**等，导致 SAM3 输出在阈值附近波动
- **排除**：遮挡（包可见）、prompt 完全无效（同视频有成功帧）

### 4. 局部遮挡 + 成像困难（necklace 单帧 frame_04）

`0038-necklace5` 仅 **frame_04** 失败，前后帧（frame_03、frame_05）均成功，prompt `"necklace"` 在同视频其他 7 帧有效。

- frame_04：手持花形吊坠，**拇指遮挡**吊坠连接处，金属/宝石有**强镜面高光**
- 项链本体仍可见，**非 prompt 问题、非全程不可见**
- 更可能是该帧**局部遮挡 + 高反光**使置信度跌破 0.1 的偶发失败

---

## 汇总判断

| 可能原因 | 是否为主要因素 | 涉及帧数 |
|----------|----------------|----------|
| Prompt 与物体类别不符（backpack vs handbag） | **是** | 8（scene03） |
| 视角/佩戴方式与 prompt 语义偏差（背对 + 斜挎） | **是** | 14（scene02 全 8 + scene01 部分 6） |
| 置信度阈值边界（同场景时成时败） | **是** | 6（scene01 失败帧）+ 1（necklace） |
| 目标被遮挡 | **否**（handbag 均可见；necklace 仅轻微局部遮挡） | 0–1 |
| 画面无目标 | **否** | 0 |

---

## 建议后续验证（可选）

1. 对 scene03 用 prompt `"backpack"` 重跑，预期可恢复 mask
2. 对 scene02 尝试 `"messenger bag"` / `"sling bag"` / 降低 `confidence_threshold`
3. 对 necklace frame_04 单独重跑或略降阈值，确认是否为偶发边界 case

重新导出本目录：

```bash
source /data/liuluyan/miniconda3/etc/profile.d/conda.sh
conda activate sam3
python /data/liuluyan/sam3/report_figures/export_no_mask_failures.py
```
