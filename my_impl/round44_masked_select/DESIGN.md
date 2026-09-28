# Round 44: MaskedSelectV3 —— 我的设计（A 节前置）

> 语义：y = x[mask]（动态输出长度 = mask 真值个数）。已读：`masked_select_v3.h`（394 行）GenerateMask/GatherResult/Compute。

## A. CHECKLIST 设计必答

1. **压缩原语（R33 的深化）**：mask(uint8 0/1) → `Cast` 升半精度 → `CompareScalar(EQ 1.0)` 生成 **bitMask（uint16/uint32 位图 tensor）** → `GatherMask(dst, src, bitMask, true, count, params, rsvdCnt)` **bit-packed 压缩**——每个 bit 选择一个源元素，rsvdCnt=输出个数。**免索引列表生成**（对比 R33 索引模式：谓词逐元素时位图直用，索引来自上游时才用索引模式）。
2. **位宽随元素宽缩放**：`IS_8/4/2/1_BYTES_TYPE` constexpr 族——int64 元素一个=8×uint32 位图（GM 侧还要 int64→int32×2 ReinterpretCast 搬运，`DataCopyPadDoubleWord`）；位图宽度=元素宽度的镜像。
3. **动态输出 shape**：输出长度=rsvdCnt（数据依赖）——tiling 不可知，**shapeout 张量运行时写入**。
4. **单 repeat 压缩**：`params{1, repeatTimes=1, src0RepeatStride=STRIDE, 1}`——压缩在一次 GatherMask 内完成（tile 级）。
