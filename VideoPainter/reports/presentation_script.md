# 组会汇报稿 - VideoPainter 灯珠矩阵问题排查

**时长：** 约 5 分钟

---

## 开场（30秒）

大家好，这周汇报 VideoPainter 复现中遇到的一个严重问题和我的排查过程。

它能基于prompt的引导，实现给视频添加/去除物体，替换物体，或者给物体变更颜色的功能。

官方展示的效果非常好（翻到周报的 teaser 图），inpaint 区域能生成自然连贯的内容。

## 问题描述（30秒）

但我在复现时，所有生成结果的 inpaint 区域都出现了这种蓝色的网格状伪影（翻到灯珠矩阵的图），看起来像 LED 灯珠阵列。背景完全正常，但物体被去掉的区域就是一团蓝色网格，完全不可用。

一开始我认为这是 RTX 5090 新卡的硬件兼容性问题，所以我这周主要做了问题的排查。

## 排查过程（2分30秒）

一共做了 7 组实验，逐个排除变量。

**第一步，换 PyTorch 版本。** 试了 2.8、2.10、2.11 三个版本，还试了禁用 cuDNN、用 torch.compile 编译——全部都有灯珠。排除 PyTorch 版本。

**第二步，换 GPU。** 在 Colab 的 A100 上跑——A100 也有灯珠。这直接推翻了之前的硬件假说。

**第三步，关闭 Branch 模型。** VideoPainter 的核心是一个 branch 模型做上下文引导，我把它的 conditioning scale 设为 0 完全关掉。结果两组输出 pixel 级完全一样，branch 根本没在起作用。排除 branch 问题。

**第四步，测纯 CogVideoX。** 最关键的一步——我把 VideoPainter 的所有组件都去掉，只用标准的 CogVideoX Image-to-Video pipeline，不涉及任何 inpainting 逻辑。结果：整帧全是灯珠。这说明问题在 CogVideoX 基础模型层面，不是 VideoPainter 的代码 bug。

**第五步，去掉 CPU offload。** 分析官方代码发现，官方用的是 `pipe.to("cuda")` 直接把所有模型放在 GPU 上，而我们因为显存只有 32GB，用了 CPU offload 让模型在 CPU 和 GPU 之间搬运。我减少到 13 帧，也用了 `pipe.to("cuda")`——灯珠还在。

**第六步，加载 LoRA 权重。** 读了官方 Gradio Demo 的代码，发现他们还加载了 VideoPainterID 的 LoRA 权重，我们之前一直没加。补上之后——灯珠还在。

**第七步，完全复刻官方代码。** 把官方 `app/utils.py` 里的模型加载和推理逻辑原封不动复制过来，所有参数一模一样——灯珠还在。

## 结论（1分钟）

7 组实验全部都有灯珠。但官方效果明明是好的。那区别在哪？

分析他们的 `requirements.txt` 和代码后发现，官方用的是 **PyTorch 2.4.0** + **80GB 显存的 A100 或 H100**。而我们的 RTX 5090 不支持 PyTorch 2.4.0（因为 Blackwell 太新），Colab 的 A100 只有 40GB 显存不够用 `pipe.to("cuda")`。

所以灯珠很可能是两个因素叠加：一是 RTX 5090 + PyTorch 2.10 的 CUDA kernel 与 CogVideoX 的 patch 机制不兼容；二是 CPU offload 机制本身与这个 pipeline 的 branch-transformer 交替调用模式不兼容。要正常运行需要同时满足正确的 PyTorch 版本和足够的显存。

## 下周计划（30秒）

1. 第一个，租一台 A100 80GB 的云服务器，用 PyTorch 2.4.0 + `pipe.to("cuda")` 做决定性测试。如果没灯珠就确认了我们的结论。

2. 第二个，在 GitHub Issues 上提问，把排查结果反馈给作者。

3. 灯珠解决后，开始在商品视频数据上做 inpainting 测试。

以上就是这周的工作，谢谢大家。
