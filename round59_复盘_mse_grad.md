# Round 59 复盘：MSELossGrad（多输入单队列打包）

> 我的实现：`my_impl/round59_mse_grad/`（DESIGN.md / mse_grad_custom.h）
> 对比对象：`loss/mse_loss_grad_v2/op_kernel/mse_loss_grad_v2.h`（KernelMseLossGrad910）

---

## 一、多输入单队列打包（本轮核心新 idiom，CHECKLIST B51）

生产把三个输入（predict/label/dout）装进**一个队列槽**：`InitBuffer(inQueueIN, bufferNum, align*3)`，一次 Alloc 后三个 DataCopyPad 写 `[0]/[align]/[2*align]` 三段，一次 EnQue/DeQue。**N 输入 = 1 次队列操作**。对照 R24 我给 dgamma/dx 开独立队列——多输入同 tile 消费时应打包单队列（队列操作省 N−1 倍），代价是槽内偏移管理。

配套细节：
- **bf16 三段一次 Cast**：`Cast(f32, packed, ..., align*3)` 对打包后整槽一次 cast（cast 粒度与打包粒度对齐）；
- fp32 段寻址 `inLocal[align]` / `inLocal[2*align]`（ReinterpretCast 偏移切片，R21 B23 同源）。

## 二、两个 tiling 机制（B7/R9 的机制化实例）

1. **usedDb 字段**：buffer 深度（1/2）host 决定，tileNum 随 bufferNum 翻倍——**BUFFER_NUM 是 host 的 tiling 输出**而非 kernel 常量（R9 的"是输出"落到具体机制）；
2. **尾核降级**：padLength≠0 的尾核切 tileNum=1/bufferNum=1/tileLength=padLength——**尾核放弃流水换简单正确**（尾块特判集中在 Init 而非循环内分支）。

## 三、grad 数学与 cof 融合

dx = (predict−label)×cof×dout，cof=2/N（mean）host 算好（C5）；三指令 Sub→Muls(cof)→Mul(dout)。与 R47 正向对偶：正向全量归约（跨核两段聚合），反向逐元素广播 cof——**loss 族正向归约/反向广播的 cof 对偶**。

## 四、CHECKLIST 增量

- **B51（新）**：多输入同 tile 消费 → 单队列打包（一次 Alloc 搬 N 输入到槽内偏移段，一次 EnQue/DeQue），bf16 打包整槽一次 Cast；usedDb tiling 字段定缓冲深度（tileNum 随之翻倍）；尾核降级单 tile（特判集中 Init）。
- **B7 机制化注**：BUFFER_NUM 由 host tiling 字段（usedDb）决定并随 tileNum 联动。

## 下一轮候选

conv2d_transpose_v2（转置卷积 dim 推导）或 R60 小元轮（loss/conv/pooling 入树）。
