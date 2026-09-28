# Round 2 复盘：Softmax 算子（设计 + 实现 全流程对比）

> 我的实现：`my_impl/round2_softmax/`（DESIGN.md / softmax_custom.cpp / op_host/）
> 对比对象：`reference/cann-ops/src/math/inplace_attn_softmax/`（生产级 softmax 家族实现，行方向 softmax，支持 fp16/bf16/fp32、大 shape）

---

## 一、设计层对比

| 设计维度 | 我的实现 | cann-ops 生产实现 |
| --- | --- | --- |
| 核心算法 | **手写三遍扫描**（读 max→读 exp 累加 sum→读 exp 除 sum），GM 上的 x 被读 **3 遍** | **调用高阶 API `SoftMax<inType>(dst, src, SoftMaxTiling, srcShape)`**，块内一次算完；x 只读 **1 遍** |
| 核内分块 | 一次一片单行（row 内 tile 列） | **二维块**：`basicRowLen × basicColLen`（一次搬多行），`DataCopyParams{blockCount, blockLen, stride}` 一步搬运，摊薄指令开销 |
| 大 shape（行装不下 UB） | 三遍扫描天然支持（代价是 3 倍带宽） | 单独的 `BigShape` 模板类 + 独立 TilingKey 分支（111/211/311），列方向再切 colLoop |
| 精度策略 | 以 T 精度逐 tile 计算，标量状态用 float，最后用 `1/sum` 乘法（有精度损失） | **fp16/bf16 一律 Cast 到 fp32 做 softmax，再 CAST_RINT 转回**（`isCast` 模板参数控制，TBuf 存 fp32 中间量） |
| 未对齐尾块 | 用普通 DataCopy 硬搬（Round 1 遗留问题没解决） | **`DataCopyPad` + `DataCopyPadParams`**：不足 32B 的尾部自动补齐 padding；同时 `SoftMaxShapeInfo{..., oriColLen}` 告诉高阶 API 真实列长，pad 区不参与计算 |
| 行数 < 核数 | host 强行 `coreNum = rowCount`（重设 blockDim） | blockDim 保持总核数，kernel 里 `if (GetBlockIdx() >= realCoreNum) return;` 早退 |
| 多维输入 | 只支持 (M, N) | **fold 成两轴**：所有前导维合并为 rowLen，只对 last dim 归约 |
| workspace | 我传了但没用 | `if (workspace == nullptr) return;` + `GetUserWorkspace(workspace)` 规范取用 |

### 设计层最大的教训：先查高阶 API，再决定手写

文档概览里白纸黑字写着"**高阶API：封装单核公共算法（如卷积、矩阵运算等）**"，我在设计 softmax 时完全没有回头翻 API 文档确认"softmax 有没有现成的高阶接口"，直接手写了三遍扫描。生产代码里，开发者要做的事情收缩为：

1. 把输入按 `basicRowLen × basicColLen` 二维切块；
2. `DataCopyPad` 搬入 → `SoftMax` 高阶 API → `DataCopyPad` 搬出；
3. host 侧用 `GetSoftMaxMinTmpSize` / `SoftMaxTilingFunc` 算出高阶 API 自己的 tiling（`SoftMaxTiling` 结构，随 TilingData 下发）。

**手写归约循环是最后手段**——只在没有匹配的高阶 API、或高阶 API 性能不满足时才做。我设计时把"教学样例用基础 API"错误泛化成了"所有算子都用基础 API 搭"。

另外两个反直觉设计决策：

- **BUFFER_NUM = 1**：softmax 的块是多行大块，double buffer 的 UB 代价超过了流水收益。Round 1 我把"BUFFER_NUM=2"当成了默认信条，实际上它是按块大小权衡的结果。
- **多行一块**：我按"行是归约原子，一次一片"直觉写成单行分片；社区按"UB 能装多少装多少行"，用 DataCopy 的 blockCount 维度把多行打包。**tiling 的原子单位是 UB 块，不是逻辑语义单位**。

---

## 二、代码层对比（对照我的 `softmax_custom.cpp`）

### P0 —— 正确性/API 使用问题

1. **TBuf 用法错误**：我写 `tmpBuf.Get<T>(0)`、`tmpBuf.Get<T>(1)` 想在一个 TBuf 里切两块——TBuf 的 `Get` 不是按偏移取子块的正确姿势，应声明两个 TBuf 或用一个大 TBuf 手动 `Get<T>()` 后自己算偏移。且 `InitBuffer(tmpBuf, BUFFER_NUM, ...)` 把 TBuf 当队列配了 double buffer，语义混乱。
2. **`ReduceMax/ReduceSum` 的签名是猜的**（我标注了 //??）：包括是否需要 workspace 参数、输出写 dst[0] 还是按 repeat 分布。真实工程里这套跨 tile 归约要么用高阶 API，要么用 `WholeReduceMax` 等专用接口——我连接口族都没选对。
3. **标量状态在 UB 上开 32B buffer 再 `GetValue(0)`**：行 max/sum 这种标量放在寄存器/标量通路即可，我在 UB 上绕了一圈，多一次 UB 读写延迟。

### P1 —— 性能问题

4. **3 倍 GM 读带宽**（见设计层）：`x` 三遍扫描重读。生产实现靠二维块 + 高阶 API 做到单遍。
5. **fp16 精度**：我逐 tile 以 fp16 做 Exp，sum 用 float 标量累加，最后乘倒数——三处精度损失。生产做法是整块 Cast fp32 计算，误差只在进出转换处。

### P2 —— 工程规范差距

6. TilingKey 没用起来：我只有一个模板类，fp16/fp32 走同一条路径；生产用 101/111/201/211/301/311 六个 key 分派 dtype × 大小 shape。
7. head/tail core 命名与 realCoreNum 早退模式（见设计层）。
8. host 侧缺少参数校验（CheckOpParams）、dtype 长度表（`x_dTypeLen[xDtype]`，我硬编码 `dataTypeLength = 2`）。
9. 我自己的低级不一致：初版 kernel Init 收 `rowsPerCore/rowOffset` 而 tiling 传的是 `smallRows/tailRows`（写完 host 才发现，回头修的）——生产实现里 kernel 消费 tiling 的字段有统一注释约定，我在 kernel 里没有维护一张 tiling 字段对照表。

### 我写对的部分

- 行为单位的多核切分 + 大小核 + 行偏移计算；"减 max 防 exp 溢出"的数值稳定设计；UB 预算表意识（6 份 × BUFFER_NUM）；uint64 长度；GET_TILING_DATA / REGISTER_TILING_DATA_CLASS / InferShape / InferDataType 工程骨架——Round 1 的教训都被吸收了，这轮 P0 里没有再出现"不整除就算错"级别的设计错误。

---

## 三、经验教训

1. **写算子前先查高阶 API 清单**（SoftMax、Reduce、Matmul、Convolve……）。Ascend C 的定位是"多层 API"：基础 API 只是没有现成算法封装时的拼装材料。设计文档第一步应该有一节"API 选型"。
2. **tiling 的原子是 UB 块，不是语义单元**。二维块（多行×多列）+ DataCopyParams 的 blockCount/stride 是通用套路，比逐行逐片高效一个量级。
3. **`DataCopyPad` 是未对齐问题的标准答案**（Round 1 遗留问题正式销案），配合 shapeInfo/padParams 保证 pad 区不污染计算。
4. **低精度输入、fp32 中间计算**是精度设计的默认解；除法优先转成乘倒数，但 sum 的累加本身要高精度。
5. **double buffer 不是信仰**：块大到一定程度，BUFFER_NUM=1 反而正确。
6. **核数分配早退优于重设 blockDim**：`GetBlockIdx() >= realCoreNum` 直接 return。
7. 对没把握的 API 不要"猜着写"：把不确定点显式标注（我用了 //??）并在对比环节逐个销案，比含糊写完强——这轮我的 //?? 清单 5 条全部命中真实差距。

## 四、Round 3 候选方向（待续）

- 候选 A：ReduceSum（纯归约，学习 WholeReduce 系列与跨核归约）；
- 候选 B：Matmul 类（Cube 通路 + Matmul 高阶 API，和矢量通路完全不同）；
- 候选 C：LeakyReLU/GELU 等激活（学习"查高阶 API 先行"能否内化）。
- 无论选哪个，第一步先写"API 选型"一节。
