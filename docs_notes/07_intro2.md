---
title: 简介
description: "AI Core是AI处理器的计算核心，一块AI处理器通常集成多个AI Core以实现并行计算。相比传统CPU，AI处理器更适用于模型训练与推理，这得益于其内部集成的大量计算单元以及丰富的向量计算指令——单个硬件指令即可完成对多组数据的并行操作。"
url: https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0018.html
sourcePath: /source/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0018.html
indexId: 1b386450a3250f3dd16ecf1d33827f1fa6c1dbecb6e4c7a0972d876cc247701384
---
# 简介

AI Core是AI处理器的计算核心，一块AI处理器通常集成多个AI Core以实现并行计算。相比传统CPU，AI处理器更适用于模型训练与推理，这得益于其内部集成的大量计算单元以及丰富的向量计算指令——单个硬件指令即可完成对多组数据的并行操作。

AI处理器上主要支持以下两种编程模型：

- SIMT（Single Instruction Multiple Thread，单指令多线程）：以单条指令驱动多个线程的形式实现并行计算。
- SIMD（Single Instruction Multiple Data，单指令多数据）：以单条指令操作多个数据的形式实现并行计算。

其中，SIMT编程适用于离散访存、复杂分支控制等场景下的矢量算子开发，也便于熟悉业界SIMT算子开发的开发者快速上手Ascend C算子开发。

SIMD编程则适用于纯矩阵计算、连续访存的矢量算子或融合算子场景。

此外，我们还提供SIMD与SIMT混合编程方式，融合两种模型的优势，在性能与易用性之间取得更好的平衡。
