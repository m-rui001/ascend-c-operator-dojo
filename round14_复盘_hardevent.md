# Round 14 复盘：HardEvent 事件体系专题

> 我的实现：`my_impl/round14_hardevent/`（DESIGN.md / event_mul_custom.cpp）——脱离 TQue 的裸流水：TBuf 双槽 + 显式事件对亲手管理 MTE2/V/MTE3 依赖
> 对比证据：API 列表同步控制四件套 + cann-ops 全库 HardEvent 频率表（4700+ 处使用）+ rms_norm/add_layer_norm 实例

---

## 一、事件体系全景（本轮知识沉淀）

**四层同步机制**（从细到粗）：
1. `PipeBarrier<PIPE_X>`：同管内指令序；
2. `SetFlag/WaitFlag<HardEvent::X_Y>`：同核**跨管**依赖（X 完成→Y 才能继续）；
3. `CrossCoreSetFlag/WaitFlag`：AIC↔AIV 跨核（Round 7）；
4. `SyncAll`：全核屏障（Round 10）。

**HardEvent 事件对 = 有向依赖边**，生产频率表即"依赖图的热边"：V_MTE3（算完才能搬出，842 次）、MTE2_V（搬入完才能算，838）、V_S（算完才能读标量，765）、**MTE3_MTE2（上一轮搬出完才能复用槽，728——双缓冲轮转的生命线）**、S_V（标量写完才能算，680）。

**FetchEventID 机制**：`GetTPipePtr()->FetchEventID(HardEvent::X_Y)` 从 TPipe 领取递增事件号——同一事件对可**多笔在途**（双缓冲下两笔 CopyIn 各领一个 id），不是全局一个 EVENT_ID0。这是我 DESIGN //?? #2 的答案：池深由 TPipe 管理，开发者只管"每笔依赖领一个号"。

## 二、我的裸流水实现自评

1. **写对了的**：三对核心依赖（MTE2_V / V_MTE3 / MTE3_MTE2）的挂点——槽复用的 `WaitFlag(MTE3_MTE2)` 放在本轮 CopyIn 前、`SetFlag` 放在上轮 CopyOut 后（//?? #1 自答：SetFlag 挂在发起 MTE3 的代码之后即可，硬件按队列完成序置位）；S_V 演示段把标量-矢量握手走通。
2. **遗留问题**：Init 里一行残留坏代码（`SetGlobalBuffer` 三参垃圾调用）写完自查才发现——已删；w 的广播场景偷懒成同布局搬运（注释标明应按 R5 行常驻）；statBuf 忘了在 InitBuffer 列表里体现（漏分配，GetValue 会踩空）——**裸流水下没有队列兜底，每个缓冲、每对事件都要自查**，这是本专题最大的心智负担，也正是 TQue 存在的意义。
3. `sv.GetValue(0)` 读标量后又用 `Muls(...标量...)`——绕一圈等价于直接 Muls(scale)，S_V 演示本身在真实代码里应该用 Muls 直传；S_V 的真实场景是"标量计算出的形状/系数写进 UB 表再被 V 消费"。

## 三、TQue 与裸事件的关系（本轮的结构性结论）

- TQue = **预制的依赖对组合**：EnQue≈SetFlag(MTE2_V 或 V_MTE3)，DeQue≈WaitFlag——外加槽管理（MTE3_MTE2 隐藏在 FreeTensor/AllocTensor 里）。Round 1-13 用的队列范式是把本轮这些事件对打包了。
- **什么时候下探裸事件**：a) 队列粒度太粗（想在 V 完成前就让 MTE3 预取下一块）；b) 队列外的常驻缓冲复用链（R12 的 PipeBarrier 链）；c) V_S/S_V 这类队列不覆盖的标量握手。
- 生产代码两种风格并存：add_layer_norm 主流程用队列，细粒度处（GetValue 前、ReinterpretCast 复用处）插裸事件——**不是二选一，是分层混用**。

## 四、CHECKLIST 增补

- B13：裸流水/TBuf 复用场景，槽轮转必须 MTE3_MTE2 事件对闭环（SetFlag 在 CopyOut 后、WaitFlag 在同槽 CopyIn 前）。
- B14：FetchEventID 每笔依赖领一个号，不复用固定 EVENT_ID0。

## 五、经验教训

1. 事件对频率表是"生产依赖图的热力图"——**学同步机制先统计真实用法分布**，比读文档枚举高效。
2. TQue 不是黑盒而是依赖对的打包；理解裸事件后，Round 1 的 EnQue/DeQue 才真正"看懂"。
3. 裸流水的自由度 = 自查负担：本轮连 statBuf 漏分配都能溜进去——**无队列兜底时，InitBuffer 清单与事件对清单要一一结对检查**（下轮起 DESIGN 里画"缓冲×事件"配对表）。

## 十四轮总览

| 轮 | 主题 | 最大盲区 |
| --- | --- | --- |
| 1-13 | （见前几轮） | — |
| 14 | HardEvent 事件体系 | TQue=依赖对打包的认知；槽复用的 MTE3_MTE2 闭环；FetchEventID 多笔在途 |

**下一轮候选**：算子线继续（按 R12 精化路线选一个独立算子实战高阶 API，如 Silu/Sigmoid）；或回到 2_features 有代码的 10_communicate_compute_fused（通信计算融合）。
