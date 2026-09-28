# Round 33 复盘：IndexSelect/Gather（索引访存新结构）

> 我的实现：`my_impl/round33_index_select/`（DESIGN.md / index_select_custom.h，双路径 mini 版）
> 对比对象：`cann-ops/src/index/gather_v3/`（base + 4 模板变体，GatherMask 实战集）

---

## 一、GatherMask 双形态（本轮核心新知）

1. **掩码位模式（maskMode=true）**：mask 是 uint32 位图——`CompareScalar(GE/LT) → And → ReinterpretCast<uint32_t> → GatherMask(..., pattern, true, n, {1,1,8,0}, rsvdCnt)`，bit=1 的元素被**压缩**到输出，`rsvdCnt` 返回保留个数。生产用它做**索引范围过滤+压缩**（CalcIdxInRange：把超界索引从索引表里 tensor 化剔除），全程无标量循环——**B3"标量通路禁令"在索引域的形态：过滤/压缩用 Compare+位图+GatherMask**。
2. **索引模式（maskMode=false）**：mask 是索引 tensor，按索引从源 UB 聚集元素——内维 gather 的主力。//?? 参数形态未完全销案（索引 dtype/单位、跨 repeat 行为），生产 tmpl_1 的用法需更深读——跨轮滚动。

另有 **VREDUCE_MASK_ALL + repeat 步长 gather** 做去交错（DeAlign），{1, lineNum, lineBlockNum, 0} 的 repeatStride 参数把规整块"抽行"——GatherMask 第三种用途：布局整理。

## 二、模板变体分类（gather_v3 的 4 模板）

按 axis（0=行选/连续 vs d=内维/离散）与索引处理方式分 tmpl_0..3——**访存连续性决定路径**：行选只需逐行 DataCopy（行连续，GatherMask 是杀鸡牛刀），内维离散才必须 GatherMask。生产甚至为 TransposeB16/B8（TransDataTo5HD）配了字节级转置——索引类算子的实现自由度远大于逐元素算子。

## 三、我的差距

1. pathA 逐行标量读索引 `GetValue` 是 O(idxNum)——生产 CalcIdxInRange 用 Compare+GatherMask 全 tensor 化（含范围过滤），我的版本未做有效性过滤（越界索引直接崩）；
2. pathB 索引模式的 GatherMask 参数是推测（//?? 未销案）；
3. 生产 base 类承载"外维分块 × 索引分块"两级循环与 4 模板分派——我只写了最简单行路径。

## 四、CHECKLIST 增量

- **B32（新）**：索引过滤/压缩 = CompareScalar + And + GatherMask(掩码位模式) + rsvdCnt，禁标量过滤循环；行选（连续）用逐行 DataCopy，内维离散才用 GatherMask 索引模式。
- **A4 补充**：索引类算子的瓶颈在**访存离散度**——连续维度直接搬，离散维度 GatherMask；布局整理可用 VREDUCE_MASK_ALL+repeatStride 的抽行形态。

## 下一轮候选

scatter 类（dynamic_quant_update_scatter / scatter_add_with_sorted——散写+原子）；或 gather_v3 索引模式深读（销案 pathB）。
