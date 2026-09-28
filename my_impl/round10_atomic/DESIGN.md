# Round 10: AtomicAdd 与跨核同步 —— 我的设计（写代码前，未 grep 社区代码）

## 0. 前置功课（samples 三个样例 README 均为空壳，文档面来自已读过的章节）

已有知识盘点：
1. **AtomicAdd**（概念）：`SetAtomicAdd<T>()` 后，MTE 搬运（DataCopy 到 GM/UB）由"覆盖"变"累加"——多核对同一 GM 地址累加无需额外归约遍。
2. **CrossCoreSetFlag<modeId, pipe>(flagId) / CrossCoreWaitFlag**（Round 7 BareMix 实证）：AIC/AIV 组内跨核同步。
3. **SyncAll**（workspace 文档）：全核同步，需要 workspace 作入参。
4. **DataStoreBarrier / DataCacheCleanAndInvalid**（API 目录 + Round 9 搜索）：内存屏障/缓存可见性控制——标量写回后立即可见的保证。

## 1. 练习算子：多核全局标量和（AtomicSum）

`z[0] = sum(x[0..totalLength))`，z 预清零，fp32。

### 两种方案的设计对比（写代码前先想清楚）

| 方案 | 流程 | 同步成本 | 适合 |
| --- | --- | --- | --- |
| A：原子累加 | 每核 WholeReduceSum 得本核部分和 → SetAtomicAdd → DataCopy 1 元素到 z[0]（硬件在 GM 上累加） | 无核间同步，MTE 原子性保证 | 部分和数量少（=核数），写冲突靠硬件原子 |
| B：两段归约 | 每核写部分和到 workspace[blockIdx] → SyncAll → 0 号核二次归约 | 一次全核同步 | 核数多/原子不支持 dtype |

选 A 为主实现（fp32 原子加通常支持），B 作为结构对照写在 DESIGN 里。

### 方案 A 的核内流程

```
chunk 按 Round 1-3 语义切核（大小核 + 32B 对齐）
每核: for tile: DataCopy in → WholeReduceSum 累计 → localSum(fp32, 1元素)
     SetAtomicAdd<float>()          // 开启 MTE 原子累加模式
     DataCopy(zGm[0], sumLocal, 1?) // //?? 1 元素搬运是否合法/需 32B 槽
     UnSetAtomicAdd()               // //?? 是否需要恢复
     DataCacheCleanAndInvalid?      // //?? GM 原子写是否需要屏障（对比环节销案）
```

## 2. 不确定点（//?? 清单）

1. SetAtomicAdd 的准确签名与"作用域"（设置后是否影响后续所有 DataCopy，直到何时结束）？
2. 原子 DataCopy 的元素数：1 元素是否合法？是否必须 32B 对齐槽？
3. fp32 GM 原子加的 dtype/芯片支持范围（AtomicAdd 产品支持矩阵，Round 4 教训）？
4. z 预清零谁做：host memset（aclrtMemset）还是 kernel 0 号核？两段式里 SyncAll 的位置？
5. 原子累加和浮点顺序性：多核累加结果与单核顺序和的差异是否可接受（deterministic 诉求时怎么办）？
6. SetFlag 的 pipe 参数（PIPE_MTE3? PIPE_FIX）与 flagId 的约定——Round 7 只见过 `CrossCoreSetFlag<0x2, PIPE_FIX>(3)` 一个实例。
