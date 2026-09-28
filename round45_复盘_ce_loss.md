# Round 45 复盘：CrossEntropyLoss（LSE + target 索引 + 批级归约）

> 我的实现：`my_impl/round45_ce_loss/`（DESIGN.md / cross_entropy_custom.h，fp32 mini）
> 对比对象：`loss/cross_entropy_loss/op_kernel/cross_entropy_loss.h`（391 行）+ base（397 行）

---

## 一、LSE 稳定 idiom（R2 softmax 的 log 域闭环）

R2 做过 softmax（减 rowMax 防 exp 溢出），交叉熵多一步 **log(Σexp)**：

```
rowMax → Σexp(x−rowMax) → logBatchSum=log(batchSum)
loss = logBatchSum − x[target]
logProb_i = x_i − rowMax − logBatchSum（log-softmax，顺带在驻留行上算出）
```

生产三扫描结构与我的 mini 一致（GetRowMax → GetExpSum → GetLogProbOut）。**logProb = log-softmax 是 LSE 的副产品**——同一次驻留行计算顺带产出，符合"中间量不落 GM/不重算"主旨。

## 二、target 标量索引与多输出

1. `targetGm.GetValue(batchIdx)`——目标类号是标量语义（B3 允许级），loss 需要驻留行内随机访问 `row.GetValue(target)`；
2. **双输出**：loss（标量/行）+ logProb（逐元素）——logProb 逐行流式写出，loss 攒批按 reduction 聚合（none/mean/sum 三模式，生产 `CalcMeanLoss/GetSumLoss/GetNoneLoss` 三函数）；
3. **label smoothing**：额外一遍扫描（Σsmooth·logProb + Ln(weight) 求和）——依赖 logProb 完成，无法流式（R39 流式原则的例外：依赖链决定扫描时机）。

## 三、与 R2 的差异点（精度与规模）

- R2 softmax 是"独立算子练习"（固定 shape、无 target/多输出）；生产 CE 是 label smoothing + 多 reduction + 多输出的完整契约；
- 生产把 cast/copy 基础设施抽到 base（397 行）——loss 族（cross_entropy_loss_grad/ctc_loss_v3_grad 共享）。

## 四、CHECKLIST 增量

- **B41（新）**：LSE 类（交叉熵/log-softmax）= rowMax 稳定 + Σexp + log，logProb 是 LSE 副产品顺带产出；target 标量索引驻留行随机访问；reduction 模式（none/mean/sum）= 攒批后批级聚合；label smoothing 依赖 logProb 完成后额外扫描（依赖链决定扫描时机）。

## 下一轮候选

CHECKLIST 对表小元轮（R40-R45 的索引/散写/loss 分支入 DECISION_FLOW）或 mse_loss_v2（回归 loss 对比）。
