# Round 14: HardEvent 事件体系专题 —— DESIGN（CHECKLIST 逐项过）

## 0. 前置盘点（文档 + 生产频率表）

**同步控制四件套**（API 列表）：SetFlag/WaitFlag（同核跨流水）、PipeBarrier（同流水内）、CrossCoreSetFlag/WaitFlag（AIC↔AIV）、SyncAll（多核）。

**生产 HardEvent 频率表**（cann-ops 全库 grep）：
| 事件对 | 次数 | 语义 |
| --- | --- | --- |
| V_MTE3 | 842 | 算完才能搬出 |
| MTE2_V | 838 | 搬入完才能算 |
| V_S | 765 | 算完才能读标量（R13 的 V_S） |
| MTE3_MTE2 | 728 | **上一轮搬出完才能复用该槽搬入**（槽复用依赖！） |
| S_V | 680 | 标量写完才能算（如 SetValue 配置系数后） |
| V_MTE2 / MTE3_V | 402/379 | 反向依赖（写后读等） |
| S_MTE3 / MTE3_S / MTE2_S / S_MTE2 | ~200 | 标量与搬运的可见性对 |

**关键机制**：`GetTPipePtr()->FetchEventID(HardEvent::X_Y)` 从 TPipe 取**递增事件号**——同一事件对可多笔在途（每笔一个 id），不是固定 EVENT_ID0。

## 1. 本轮练习：事件驱动版 Mul（脱离 TQue 的裸流水）

用 TBuf + 显式事件对实现双槽 ping-pong，亲手管理三管依赖：

```
槽 s = i % 2
[i>=2] WaitFlag(MTE3_MTE2, slot_s)     # 槽复用：上轮搬出完
CopyIn:  DataCopy(xBuf[s], yBuf[s])    # MTE2
SetFlag(MTE2_V, ev) ; WaitFlag(MTE2_V, ev)
Compute: Mul(yBuf_s, xBuf_s, wBuf_s)   # V
SetFlag(V_MTE3, ev) ; WaitFlag(V_MTE3, ev)
CopyOut: DataCopy(yGm, yBuf[s])        # MTE3
SetFlag(MTE3_MTE2, slot_s)             # 释放槽
```

新增 S_V 场景：scale 标量 SetValue 后再参与计算。

## A/B/C 检查单

- A1：无高阶 API（手写流水是本题目的）✓；A4：三管全手动，依赖即事件对 ✓；A7：双槽×(x,y,w,y) + 标量槽 ✓
- B1 四件 ✓（沿用 R12 模式）；B2 尾块 DataCopyPad ✓；B9/B10 展开为显式事件链 ✓；B11 N/A（无可选特性）
- C1-C5 沿用 R12 模板 ✓

## //?? 清单

1. MTE3_MTE2 的 SetFlag 应该挂在 CopyOut 之后（MTE3 完成即置位）——位置对吗？
2. FetchEventID 的池深（同一事件对最多几笔在途）？
3. 裸流水下 AllocTensor/FreeTensor 不存在，InitBuffer(TBuf) 大小即槽大小——双槽的正确布局（一个 TBuf 两倍大 + 偏移 vs 两个 TBuf）？
4. S_V 的 SetFlag 挂在 SetValue 之后、WaitFlag 在 Mul 之前——S 管与 V 管的 flag 语义确认。
