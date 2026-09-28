# Round 36 复盘：TopKV3（proposal 打包 + 硬件排序/归并）

> 我的实现：`my_impl/round36_topk/`（DESIGN.md / topk_custom.h，结构复述 mini）
> 对比对象：`index/top_k_v3/op_kernel/top_kv3.h`（315 行单类）

---

## 一、TopK 的算法结构（本轮核心新知）

TopK 不是"排序再取前 k"，而是三段硬件化流水：

1. **proposal bit-packing（编码）**：把 (score, index) 打包成可整体比较的元素——fp16 score 与 16bit 索引拼合（GatherMask `Low16Pattern/High16Pattern`（1=取第一个/2=取第二个）从 fp16 对里抽半段 + `ProposalConcat` 按槽位合入）。**排序一次带出 (value, index) 对**，免去排序后回查索引。
2. **块内排序 + MrgSort4 四路归并**：每块（ubFactor，16 对齐，pad 负无穷）排成有序 proposal；块间用 **`MrgSort4`**（硬件四路归并指令，`MrgSortSrcList`+`MrgSort4Info` 描述源与长度）归并，**ping-pong 双缓冲**交替；`ifExhaustedSuspension` 旗标在 k<16 时让耗尽的源提前暂停（归并只保留 top-k 长度）。
3. **解码写出**：`DecodeValuesFromProposal/DecodeIdxFromProposal` 拆包。

## 二、周边新知

1. **`CreateVecIndex`**：生成 0..n-1 索引向量的 API（索引构造不再手写循环）；
2. **largest==0**：`Muls(-1)` 取负——最小 TopK 复用最大 TopK 的一切逻辑；
3. **pad 负无穷用 SetValue 标量循环**：pad 元素少（<16 个），标量填充正当（B3 控制流/极小量边界），且 V_S/S_V 事件对成对出现；
4. `ifExhaustedSuspension`：硬件归并指令的可配置暂停——**归并只需 top-k 长度时，不必归并全部**。

## 三、CHECKLIST 增量

- **B35（新）**：TopK/选择类 = 打包编码（score+index 合一可比较元素）→ 块内排序 → MrgSort4 四路归并（ping-pong）→ 解码；pad 负无穷少量标量填充正当；最小化用取负复用。

## 下一轮候选

round_common.h（R33/R34 共用的归约公共件精读——把散落的归约 idiom 收拢）或插入方法轮（CHECKLIST 重排为决策树）。
