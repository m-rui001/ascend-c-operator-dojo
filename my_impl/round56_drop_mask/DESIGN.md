# Round 56: Dropout Mask Adapter —— 我的设计（A 节前置）

> 场景：FA 的 dropout 掩码以 **1 bit/元素**存 GM（省 8 倍带宽），使用前需解包为 bool/byte。random 目录无实现（空壳），以 cann-ops-adv 的 drop_mask_adapter（184 行）为对比源。

## A. CHECKLIST 设计必答

1. **位解包原语（R44 位图压缩的对偶）**：GM 位图 → DataCopyPad 进 UB（x/8 字节）→ **`Select`**（src0=1.0 常量 `Duplicate`，src1=位图且 **src1RepStride=0** 重复广播，dstRepStride=8）→ bool/fp16 展开写回。压缩（R44 Compare→位图→GatherMask）与解包（Select）是位掩码的对偶双向。
2. **空间账本（生产注释）**：x 元素 = GM x/8 字节 + UB 2x×2(fp16 select 源/结果) + GM x 字节——**位存储省 GM 带宽，付 UB 与转换指令**。
3. **Adapter 类模式**：DropMaskAdapter 是**算子附件类**（独立头，主算子构造调用，自持队列与 SyncAllCores）——非工厂、非独立算子，第三种组织形态。
4. **多核**：multiCoreFactorSize 切分 + 单核空段早退 + `SyncAllCores()` 尾同步（对齐分支不需要）。
5. **repeat 参数妙用**：src1RepStride=0（位图反复读）+ dstRepStride=8（输出逐 256B 推进）——B36 跨行 repeat-Add 的解包版。

## 1. 我的实现（Select 位解包 mini）
