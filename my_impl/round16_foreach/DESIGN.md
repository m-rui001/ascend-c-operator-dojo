# Round 16: foreach_add_scalar —— 我的设计（写代码前，未看社区实现）

## 0. 算子语义（来自 aclnn 文档）

- 输入：张量列表 x[N]（各自 shape 可不同、非连续、≤8 维，dtype fp32/fp16/bf16/int32）+ 标量 scalar（Host 侧 tensor！不是 attr）
- 输出：y[i] = x[i] + scalar
- 新结构类：**多张量批量**——单次 launch 处理一串 shape 各异的张量，省 N 次 launch 调度开销

## 1. API 选型

- 计算本体就是 Adds（基础 API，Round 4 已验证），无高阶封装可查 → 重点是**列表结构怎么进 kernel**
- dtype 分发：TilingKey × 模板（Round 2/4 教训）

## 2. 我的核心设计问题：变长张量列表的三层处理

1. **描述表传递**：N 个张量的 {GM 地址、元素数} 必须进 kernel。方案 A：tiling 里放数组（定长上限）；方案 B：放 workspace（GM），tiling 传 N 与指向 workspace 的描述表。我选 **B**（N 可变，tiling 定长数组会浪费/截断），//?? 验证社区做法。
2. **多核分配**：把列表"展平"成总元素流？不行——张量边界不对齐 32B 会让跨张量的块不合法。我的方案：**按张量顺序做贪心块分配**：维护全局块序号，每张量的块数 = ceil(len_i/ubTile)，核 j 负责块区间 [start_j, end_j)，块可以跨张量但 DataCopy 按"块所属张量内偏移"逐块搬（每块一次 DataCopy，只处理所属张量）。
3. **非连续支持**：非连续 x 的 stride 处理复杂——我的设计先只支持连续（stride 逐维折叠），非连续列入 //?? 看社区（大概率 host 侧判断非连续则转置/拷贝或拒绝）。

## 3. kernel 骨架

```
kernel(tiling: N, workspace: [{addr, count}...]):
  我的块区间 = splitByBlocks(...)
  for blk in myRange:
    (tensorIdx, inTensorOffset) = locate(blk)
    DataCopyPad(in, tensorAddrs[tensorIdx] + off, count)   // 尾块 pad
    Adds(out, in, scalar, count)
    DataCopyPad(out GM, out, count)
```

标量是 **Device 侧 tensor**（不是 tiling 常量）：kernel 需先把它读进 UB/标量寄存器一次（//?? 标量 tensor 怎么读——GetValue 有 V_S 事件对要求，CHECKLIST B10）。

## 4. UB 预算

in 1 份（BUFFER_NUM=2）+ out 1 份（2）= 4 份 ubTile。ubTile 由 UB/4 反推。

## 5. 不确定点（//?? 清单）

1. 描述表进 workspace 还是 tiling？社区结构？
2. 多核-张量映射：社区是展平流、按张量切核、还是贪心块？
3. 标量 Device tensor 的读取路径（标量在 GM 上，怎么进 kernel——DataCopy 32B？GetValue+事件对？）
4. 非连续 tensor 的处理层级（host 校验拒绝？还是 stride 化 DataCopy？）
5. dtype 分发用 TilingKey 还是模板参数由 GET_TILING_DATA 的 dtype 字段 if-else？（bf16 时 scalar 是 fp32，精度路径）
6. foreach 系列的公共代码在 cann-ops 是共享头还是每算子复制？（工程组织）
