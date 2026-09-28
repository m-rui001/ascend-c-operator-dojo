# Round 45: CrossEntropyLoss —— 我的设计（A 节前置）

> 语义（简化核心）：每行 loss = logBatchSum − x[target]，logBatchSum = log(Σexp(x−rowMax))；reduction = none/mean/sum；label smoothing 可选。
> 已读：`cross_entropy_loss.h`（391 行）Process/GetRowMax/GetExpSum/GetLogProbOut。

## A. CHECKLIST 设计必答

1. **LSE 稳定 idiom（R2 softmax 的 log 域版本）**：rowMax（MIN_FLT 初始化）→ Σexp(x−rowMax) → `logBatchSum = log(batchSum)` → `loss = logBatchSum − x[target]`——**减 rowMax 的稳定化对 log(Σexp) 同样必要**。
2. **target 标量索引**：`targetGm.GetValue(batchIdx)` 取目标类号 → loss 需要 x[target]（行内随机访问一个元素）——用驻留行 + 偏移 GetValue（B3 索引语义允许）。
3. **批级归约模式**：per-row loss 攒批后按 reduction（none/mean/sum）聚合——none 直接写，mean/sum 用标量累加（行数=批大小，O(batch) 允许级）。
4. **label smoothing**：额外一遍扫描（Σsmooth·logProb），Ln(weight) 求和——多输出附加扫描（R39 流式原则的例外：smoothing 依赖 logProb，无法首扫流式）。
5. **多输出**：loss + logProb（+workspace lse）。

## 1. 我的实现（fp32 mini：LSE + target loss + mean reduction）

逐行：载入行 → rowMax → Σexp → logBatchSum → loss = logBatchSum − x[target]；loss 攒批 → mean 归约。
