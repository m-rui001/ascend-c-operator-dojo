# Round 36: TopKV3 —— 我的设计（A 节前置）

> 语义：每行取 TopK（values+indices）。已读：`top_kv3.h`（315 行）全流程。
> 算法：**proposal bit-packing + 硬件排序 + MrgSort4 四路归并**。

## A. CHECKLIST 设计必答

1. **算法结构（本轮核心）**：
   - **proposal 编码**：把 (value, index) 打包成 32bit 可整体比较的元素（fp16 score 高位 + index 低位，GatherMask Low16/High16 Pattern 抽半 + ProposalConcat 合入）——**排序一次带出配对索引**，免得排序后回查；
   - **块内排序**：`ProposalConcat(proposal, x, rep, 4)` 合入 score 后由排序指令排（PROPOSAL_NUM_PER_REP 对齐，pad 负无穷）；
   - **MrgSort4 归并**：四路已排序 proposal 归并（MrgSortSrcList/MrgSort4Info），ping-pong 双缓冲交替；`ifExhaustedSuspension` 旗标在 k<16 时让耗尽的源提前暂停；
   - **解码**：DecodeValues/IdxFromProposal 拆包写出。
2. **largest==0**：`Muls(-1)` 取负转最大问题（最小 TopK 复用同一排序）。
3. **pad 负无穷**：`SetValue` 标量循环填（少量 pad 元素，标量正当——B3 控制流边界）。
4. **新 API**：`CreateVecIndex`（生成 0..n-1 索引向量）。

## 1. 我的实现（结构复述 mini）

Encode(打包) → 块排序 → MrgSortCustom 归并循环 → Decode。缓冲：proposalBuf（块）、proposalTopkDoubleBuf（归并 ping-pong）、proposalOutBuf（行×k）。
