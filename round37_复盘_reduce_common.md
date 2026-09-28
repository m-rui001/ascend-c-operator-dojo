# Round 37 复盘：reduce_common.h 精读（归约 idiom 收拢对表）

> 对比对象：`rms_norm_grad/op_kernel/reduce_common.h`（186 行）+ `add_rms_norm/op_kernel/reduce_common.h`（207 行）
> 本轮性质：收拢轮——把散落 R3/R21/R22/R26 的归约 idiom 对表成"决策表"。

---

## 一、公共件的五个函数与各自适用域

| 函数 | 机制 | 适用 |
| --- | --- | --- |
| `ReduceSumForSmallReduceDim` + PreRepeat | **跨行分块累加**：按 64 元素列块循环，`Add(addSum, src[col], addSum, elemNum, repeat=rows, {1,1,1,blk,repStride,blk})` 用 repeat 让**所有行同时**累加进 addSum 的对应列块；全部列块完成后按行 `WholeReduceSum` 收口 | (N, D)→(N,1) 且 **D 小**：repeat 维吃行数 |
| `ReduceSumMultiN` | 上者的入口封装（Duplicate 清零 addSum + 参数推导） | 同上 |
| `ReduceSumHalfInterval` | **折半树形求和**（R26 B29）：`Add(self, self[body], tail)` 减半迭代 + 单次 WholeReduceSum 收口 | **单行 D 大**：降低舍入误差 |
| `findPowerTwo` | 位技巧求 ≤count 的最大 2 幂（`count|=(count>>k)` 五连 + `(count+1)>>1`） | 树形求和的边界计算 |

## 二、归约决策表（收拢 R3/R21/R22/R26/R37）

写归约代码前按序自问：

1. **归约维大小 D**：
   - D 小（行多行短）→ `ReduceSumForSmallReduceDim` 跨行分块累加（repeat 吃行数）**或** R22 的 UB 内转置（DoTranspose 凑归约宽度）——两条路线按布局代价选；
   - D 大（单行长）→ 分段读入 + 折半树形求和（R26，精度优先）或普通分段 ReduceSum（常规精度）；
   - D ≤ 单 repeat 宽度 → 单次 WholeReduceSum（R3）。
2. **行数 N**：多行块驻留 → 逐行循环归约 + SetMaskCount 直读累加器（R21 B10）；整矩阵装不下 → R22 块状两遍。
3. **精度敏感** → 折半树形 / fp32 中间量（B8）；常规 → 直接归约。
4. **归约结果的消费形态**：标量（B10 累加器直读）/ tensor（B3 保持 tensor 通路）/ 跨核（R25 B28 三态）。

## 三、跨行分块累加的参数解读（B36 新知）

`Add(addSum, src[elemIdx], addSum, elemNum, repeat=rows, {1,1,1,ELEM_PER_BLK,repStride,ELEM_PER_BLK})`：
- `repeat=行数`：一次调用处理所有行；
- `srcRepStride=repStride=D_align/8`：迭代间步长 = 行跨度——**每"迭代"跳一行**；
- `dstRepStride=ELEM_PER_BLK(=8)`：addSum 内每次迭代写下一个 64B 列块；
- 循环变量 elemIdx 步进 64：逐列块推进。
即"**列块 × 行**的双层循环被 repeat 参数压成单层"，每行归约宽度不足的问题由跨行 repeat 摊薄。这是 R33 GatherMask 抽行、R22 转置之外的第三种短行处理法。

## 四、CHECKLIST 增量

- **B36（新）**：many-short-rows 批归约三路线——跨行 repeat-Add 分块累加（reduce_common 首选）/ UB 内转置（R22）/ 逐行循环（行数少时）；选择看布局代价与行数。
- **B10 场景对表**：本文件未用 SetMaskCount——`WholeReduceSum(dst, addSum, MASK_PLACEHOLDER, repeat, 1, 1, blk)` 一次出 N 行结果，行级批量归约的结果天然 tensor 化，无需逐行标量读。

## 五、元观察

reduce_common.h 在 cann-ops 中被多个 norm 算子**逐目录复制**（rms_norm_grad 186 行 vs add_rms_norm 207 行，内容小异）而非引用公共库——生产代码也存在"复制式复用"，与 foreach 的统一公共头（common/inc/foreach）不同。**公共件的归属层级是演进而非设计**：foreach 先有公共层，norm 系还在复制阶段。

## 下一轮候选

CHECKLIST 决策树化（元轮，把 37 轮教训整理成"算子开发决策流程图"文档）或 norm/add_layer_norm（mean+var 的两遍在 base 复用下的形态）。
