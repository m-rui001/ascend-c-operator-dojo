# Round 24: RmsNormGrad —— 我的设计（A 节前置）

> 反向公式（aclnn 文档）：
> - `dx_i = (dy_i·g_i − (x_i/Rms)·Mean(y))·(1/Rms)`，`Mean(y) = mean(dy·g·x/Rms)`，Rms 用正向传入的 rstd
> - `dg_i = (x_i/Rms)·dy_i`（跨行累积）
> 输入 dy/x/rstd/gamma，输出 dx/dgamma。生产 6 变体：whole_reduce_n / whole_reduce_d / split_n / split_d + high_precision×2。

## A. CHECKLIST 设计必答

1. **API 选型**：无高阶封装；变体族按"归约轴 N/D × 整行/分段 × 精度"分派（TilingKey）。
2. **我实现的变体**：whole_reduce_n 简化版（整行驻留、fp32 输入），变体族全景进复盘。
3. **封装边界**：dgamma 是**跨行归约**——行循环内 UB 累加（Add 进同 buffer），循环结束一次写出（B5 攒批的"累加"变体）。
4. **瓶颈**：读 dy/x 两遍量级、写 dx——访存型；mean 项需先归约再广播（两阶段于行内）。
5. **精度**：fp16 → fp32 中间量三份（dy/x/norm 积）；high_precision 变体存在说明默认版已有精度取舍（对比环节看差异）。

## 1. 我的实现设计（fp32 整行驻留）

```
块 = 多行 × rowAlign：
  Pass1 逐行：
    dgRow = x*rstd*dy          (Mul 两连)
    meanY = ΣdgRow / n          (WholeReduceSum + GetAccVal)
    dxRow = (dy*g − x*rstd*meanY) * rstd
    dgammaBuf += dgRow          (跨行 UB 累加——本轮新点)
  循环外：dgammaBuf 一次写出
```

## 2. //?? 清单

1. 生产的"二叉树 Add 跨段归约"（Add(dySum[i], dySum[i], dySum[i+half])）——比我 WholeReduceSum 直归约好在哪（预测：colSplitNum 分段与 256B 宽度对齐）。
2. dgamma 跨行累加的生产形态（buffer Add 逐行 vs 攒最后）。
3. high_precision 变体与默认版的实际差异（fp64？Kahan？或只是两次归约）。
4. gamma 是否需要读（dx 公式含 g，dg 公式含 g——生产 Init 无 gamma 参数！注意：此算子的 g 与 dy 融合传入？核对——Init(dy, x, rstd, gamma, dx, dgamma) 有 gamma）。
