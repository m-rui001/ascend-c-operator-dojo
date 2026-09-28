# Round 30: foreach_copy —— 我的设计（A 节前置）

> 语义：张量列表逐元素拷贝（12 种 dtype 的 TilingKey 扇出）。独立类 186 行，非工厂派生。

## A. CHECKLIST 设计必答

1. **API 选型**：无计算 → 无工厂 Compute 钩子可用（工厂为"计算形"算子设计，退化流水走独立类）；**无 VECOUT 队列**——从 dataQueue 的 local 直接 DataCopyPad 出 GM（≈ TQueBind 的手工等价：VECIN 缓冲兼任输出）。
2. **流水形态**：BUFFER_NUM=1 + 单队列（搬运瓶颈型最简形态，B7 输出验证第九例）。
3. **dtype 覆盖**：12 个 TilingKey（fp16/fp32/bf16/int8/uint8/int32/uint32/uint64/double/bool…）——全 dtype 覆盖的纯搬运是 TilingKey 扇出的极端案例。
4. **尾块**：DataCopyPadExtParams{false,0,0,0} 字节粒度（同前几轮）。

## 1. 我的实现设计

- 单队列 + 直出：CopyIn（pad 语义尾块）→ EnQue/DeQue → ComputeAndCopyOut（直接写 out GM）→ FreeTensor；
- 每核张量区间 + 均匀分段沿用 R16 骨架；
- 对比环节验证：生产是否也省掉 VECOUT 队列、TQueBind 为何未用。
