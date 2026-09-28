# Round 59: MSELossGrad —— 我的设计（A 节前置）

> 语义：dx = (predict − label) × cof × dout，cof=2/N（mean 模式）host 算。逐元素无归约。
> 已读：`mse_loss_grad_v2.h`（KernelMseLossGrad910，66-130 行）。

## A. CHECKLIST 设计必答

1. **多输入单队列打包（本轮核心新 idiom）**：三个输入（predict/label/dout）共享**一个队列槽**——`InitBuffer(inQueueIN, bufferNum, align*3)`，一次 Alloc，三个 DataCopyPad 写 `[0]/[align]/[2*align]` 三段，一次 EnQue/DeQue。**N 输入 = 1 次队列操作**（对比我 R24 的三队列三倍队列操作）。
2. **bf16 三段一次 Cast**：`Cast(f32, packed, ..., align*3)` 对打包后的三段一次 cast——打包与 cast 粒度对齐。
3. **usedDb 是 tiling 字段**：buffer 深度（1/2）由 host tiling 决定，tileNum 随 bufferNum 翻倍——**BUFFER_NUM 是 host 输出**（B7/R9）的具体机制。
4. **尾核降级**：padLength≠0 的尾核切到 tileNum=1/bufferNum=1/tileLength=padLength——尾核放弃流水换简单正确。
5. **cof 融合**：Sub → Muls(cof) → Mul(dout) 三指令；cof host 算（C5）。
