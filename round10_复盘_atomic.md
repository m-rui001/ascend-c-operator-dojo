# Round 10 复盘：AtomicAdd 与跨核同步

> 我的实现：`my_impl/round10_atomic/`（DESIGN.md / atomic_sum_custom.cpp）——AtomicSum：每核 WholeReduceSum → SetAtomicAdd → DataCopy 累加到 z[0]
> 对比对象（samples 5/6/7 全是空壳，按第 9 轮换源规律改用生产库）：
> - `cann-ops/src/math/histogram_v2/`（原子累加的真实场景）
> - `cann-ops/src/math/icamax/`（SyncAll 两段归约的真实场景）
> - `cann-ops/src/activation/ge_glu_grad_v2/`、`conv/conv2d_backprop_filter_v3/`（原子梯度累加）

---

## 一、//?? 销案结果

| //?? | 我的猜测 | 生产证据 |
| --- | --- | --- |
| 1. SetAtomicAdd 签名/作用域 | `SetAtomicAdd<T>()` + 需要恢复接口 | ✅ 完全正确：`SetAtomicAdd<T>()` … `SetAtomicNone()` 成对出现（GLU grad、histogram、conv backprop 全部如此） |
| 2. 原子搬运的元素数 | 1 元素标量累加 | ❌ **设计品味错误**：生产里原子累加全是**整块对齐搬运**（`copyLen = CeilAlignA2B(dataCount, perBlockCount)`，histogram 一次累加一串 bin）——**标量级跨核汇总不用原子**，用两段归约（见下） |
| 3. fp32 原子支持 | 待确认 | conv3d_backprop_filter_v2 用 `SetAtomicAdd<float>()`（dav_v220），fp32 在该架构可用；histogram 用 int32 |
| 5. 浮点顺序性/确定性 | 提出疑问 | **跨轮闭环**（Round 6）：conv backprop 敢用原子（顺序不定可接受）；而 cann-ops matmul 专门有 `deterministic_splitk` 变体——**需要确定性就不能原子**，两种路径在生产库并存，host 按需求路由 |
| 6. SetFlag 的 pipe 参数 | 只见过一个实例 | 补全了完整的**事件同步 idiom**（见下） |

## 二、本轮最大的设计纠正：原子 vs 两段归约的分工

生产代码的分工非常明确，我的方案 A（原子加标量）不在生产惯用法里：

- **原子累加**适合"每核产出**一串**数据、目标地址分散"的批量累加：histogram 的 bin 计数（int32）、conv backprop 的梯度块、GLU grad 的 dX 块。原子省掉的是"先写 workspace 再归约"的搬运量，代价是顺序不定。
- **两段归约**适合"每核产出**一个（或极少量）**标量"：icamax 的流程正是我 DESIGN 里的方案 B——每核 `getCoreTmpReduResult()` → `CopyTmpRstToWkGM()`（部分结果写 workspace）→ **`SyncAll()`** → **仅 `blockIdx == 0`** 二次归约并写出，且单核场景有直接跳过分同步的捷径分支。
- 判据一句话：**累加的"宽度"大用原子（省搬运），宽度小用两段（省原子开销+可确定性）**。

## 三、学到的同步接口全景（本轮知识增量）

1. **SyncAll() 两种形态**：无参（框架托管 workspace）与 `SyncAll(syncWorkspaceGm, ubSyncWorkspace, coreNum)`（显式传入，histogram 用法）。
2. **细粒度事件同步 idiom**（GLU grad 实证）：
   ```cpp
   PipeBarrier<PIPE_V>();                                          // 管内屏障：保证 V 指令落盘
   event_t ev = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);    // 取事件号
   set_flag(PIPE_V, PIPE_MTE3, ev);                                // V 侧置 flag
   wait_flag(PIPE_V, PIPE_MTE3, ev);                               // MTE3 侧等 flag
   ```
   ——这补全了 Round 9 流水调优缺失的一层：**队列间握手不只有 TQue 的隐式同步，还有显式 event 对**，用于自定义流水（如 V 算完立刻让 MTE3 搬走，不等 EnQue/DeQue 的粗粒度节拍）。
3. `PipeBarrier<PIPE_X>()`：管内指令屏障，与跨核 flag 是不同层级。

## 四、我的实现缺陷清单

1. 方案 A 选型错误（原子加标量不在生产惯用法）——但 SetAtomicAdd/SetAtomicNone 的 API 用法猜对，说明**知识对、品味错**：知道接口怎么用，不代表知道何时用。
2. 原子 DataCopy 传 1 元素：即使可行也浪费一次 32B 槽；生产对齐块的做法才是正解。
3. 尾块 DataCopy 未做 pad（沿用了 Round 3 就该销案的隐患）——连续三轮在同一处留尾巴，列入"必检清单"第 5 项：**所有 DataCopy 逐个过对齐检查**。
4. host 侧 z 预清零没有落实（方案 A 的前提），DESIGN 里提了但代码没写——设计到实现的 drop。

## 五、经验教训

1. **接口正确 ≠ 用法正确**：原子操作的知识点猜对了，但"什么规模的问题配什么机制"的判断错了。生产代码的分工（原子=批量、两段=标量）只能从读代码里学，文档不会写。
2. **确定性与原子互斥**是贯穿性约束（Round 6 deterministic_splitk ↔ 本轮 conv backprop 原子），host 按"是否需要确定性"路由是 tiling 的又一项职责。
3. 事件对（set_flag/wait_flag + HardEvent）是自定义流水的基本工具，TQue 只是它的封装——要突破范式框框做性能时必须下探到这一层。
4. 十轮了仍在犯"设计提了、代码没落"的 drop：DESIGN 的每个条目应该有代码里的对应物，复盘时逐条勾销。

## 十轮总览

| 轮 | 主题 | 最大盲区 |
| --- | --- | --- |
| 1-9 | （见前几轮复盘） | — |
| 10 | AtomicAdd/跨核同步 | **接口对、品味错**：原子配批量、两段归约配标量；事件对是自定义流水的底层工具 |

**下一轮候选**：LayerNorm/RmsNorm 复合算子（cann-ops norm 目录证据充足，可检验十轮积累的综合运用）；或循环中间小结——把十轮教训合并成一份 checklist 文档后再继续。
