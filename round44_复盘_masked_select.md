# Round 44 复盘：MaskedSelectV3（bit-packed 压缩 + 动态输出 shape）

> 我的实现：`my_impl/round44_masked_select/`（DESIGN.md / masked_select_custom.h）
> 对比对象：`index/masked_select_v3/op_kernel/masked_select_v3.h`（394 行）

---

## 一、bit-packed 压缩（R33 GatherMask 的深化）

生产压缩链：`mask(uint8 0/1) → Cast 升精度 → CompareScalar(EQ 1.0) → bitMask(uint16/uint32 位图) → GatherMask(dst, src, bitMask, true, count, params, rsvdCnt)`。

与 R33 索引模式的分工判定：

| 选择谓词形态 | 压缩原语 |
| --- | --- |
| **逐元素谓词**（mask/比较结果） | bit-packed 位图 GatherMask——免索引列表生成 |
| **索引列表来自上游** | 索引模式 GatherMask（mask=索引 tensor） |

`rsvdCnt` 即动态输出长度——**压缩与长度统计一条指令完成**。

## 二、动态输出 shape 的处理

输出长度 = rsvdCnt（数据依赖），tiling 不可知。生产：y 缓冲按**上界预分配**，实际写出用 `DataCopyExtParams{1, rsvdCnt*sizeof(T)}` 字节粒度收口，**shapeout 张量运行时写入**（下游以 shape tensor 读实际长度）。这是"动态 shape 输出"的标准契约：预分配上界 + 字节精确写出 + shape 张量回传。

## 三、元素宽度缩放的位图（IS_8/4/2/1_BYTES_TYPE）

位图宽度 = 元素宽度的镜像：fp32 元素 1:1（1 bit/元素），int64 元素 1:8（生产还要 GM 侧 int64→int32×2 的 ReinterpretCast 搬运拆读）。**constexpr 元素宽度族 + GM 视图重解释**是宽 dtype 支持的模板化方案。

## 四、我的差距

1. Compare 输出到位图的视角转换（fp32 比较结果 → uint32 位图）我用了 ReinterpretCast 示意——生产对 uint8 mask 先 Cast 到 half/int16 再 ShiftLeft+Add 组合位平面（int64 路径），位语义比我想的精细；
2. 我借 yLocal 做 maskF32 比较位——缓冲配对表内自查过（y 未写前借用），但生产用独立 maskCastBuf——B14 纪律应优先于省缓冲；
3. 多 tile 压缩的段间衔接（前缀和跨 tile）未展开——生产单文件内 rsvdCnt 累计跨段。

## 五、CHECKLIST 增量

- **B40（新）**：谓词压缩 = CompareScalar→位图→bit-packed GatherMask（rsvdCnt=输出长度）；索引来自上游才用索引模式；动态输出 shape = 上界预分配 + 字节精确写出 + shape 张量回传；位图宽度随元素宽度缩放（int64 需 GM 视图拆读）。

## 下一轮候选

CHECKLIST A/E 节对表小元轮或 cann-ops `loss` 目录（交叉熵类——多级归约+log）。
