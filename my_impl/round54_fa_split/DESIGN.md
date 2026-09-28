# Round 54: FA 变体多核切分对比 —— 我的设计（A 节前置）

> 对比对象：`flash_attention_score_s1s2_bn2gs1.h`（S1×S2 优先）vs `flash_attention_score_bn2gs1s2_b.h`（B×N2 优先）。

## A. CHECKLIST 设计必答

1. **两变体的轴优先序**（multiCoreInnerIdx 的线性分解序不同）：
   - s1s2_bn2gs1：multiCoreInnerIdx → (B, N2, G, **S1 外 × S2 内**)——S1×S2 在内层展开，sparse 三角掩码可跳块；
   - bn2gs1s2_b：`boIdx = multiCoreInnerIdx` 直映射 **B 外层**，核内再扫 G×S1×S2——B/N2 粒度匹配 KV cache 数据局部性。
2. **尾部空任务**：`multiCoreInnerLimit += 2`——**追加 2 个空任务排空三级流水**（extraInfo[3] 槽的排空代价），任务数=有效任务+流水深度。
3. **切分策略选择判据**：数据局部性（KV 复用）vs 掩码跳块收益 vs 负载均衡（B 大时 B 优先天然均衡；S 长 sparse 时 S1×S2 优先可跳块）。
4. **线性任务号 → 多维轴索引**：通用化 R17 CalcOffset 的轴分解（div/mod 链），轴序即切分策略。

## 1. 我的实现

`AxisDecomposer`：把线性任务号按可配置轴序分解为 (B,N2,G,S1,S2)——两种变体的切分即两个轴序配置；对比生产各自的映射。
