# Round 4 复盘：LeakyReLU 算子（设计 + 实现 全流程对比）

> 我的实现：`my_impl/round4_leakyrelu/`（DESIGN.md / leaky_relu_custom.cpp / op_host/）
> 对比对象：`reference/samples/operator/ascendc/0_introduction/9_leakyrelu_frameworklaunch/LeakyReluCustom/`
> 本轮新增前置功课：读官方《通用说明和约束》+《高维切分API》+ LeakyRelu 接口规范（`docs_notes/api_*.md`、`09_repeat_stride.md`）

---

## 一、本轮前置功课的收获（Round 3 行动项兑现）

1. **更正 Round 3 的一个错误**：矢量单次迭代是 8 datablock × 32B = **256 字节**（我上轮复盘写成"256bit 指令宽度"）。half 单迭代最多 128 元素、float 64 元素，`mask = 256/sizeof(T)`。
2. `repeatStride=8`（迭代间连续）、`dataBlockStride=1`（迭代内连续）是默认连续布局；`repeatTime ≤ 255`；repeatStride=0 可反复读同一块（广播求值场景）。
3. "前 n 个数据计算"形态接口（`count` 参数）内部自动设置 mask/repeat/stride——**计算侧尾块天然安全**，只有搬运侧（DataCopy）需要对齐操心。

## 二、设计层对比

| 设计维度 | 我的实现 | samples 样例 |
| --- | --- | --- |
| **计算方式** | 直接调用基础 API `LeakyRelu(dst, src, slope, count)`（1 条指令语义） | **用 Maxs+Mins+Muls+Add 四条指令拼装**：`max(x,0) + slope*min(x,0)` |
| shape/dtype 泛化 | 任意 shape fold 一维、fp16/fp32 模板 | 固定 float、教学化均分 |
| 核间切分 | 32B 块均分 + 大小核 + 每核至少 32B 上限 | 固定 BLOCK_DIM 均分 |
| UB 反推 | ubSize/4 份反推 ubTile | tileNum=16 写死 |
| attr 处理 | attr(negative_slope) → TilingFunc 读出 → tiling 下发 → kernel 传给 API | 完全一致（`GetAttrPointer<float>(0)`、`Attr("negative_slope").Optional().Float(0.0)`） |

### 本轮最重要的发现：API 的产品支持矩阵是选型的一部分

样例不用 `LeakyRelu` API 而手工拼四条，原因藏在我读的接口规范里：**LeakyRelu API 的产品支持表写着"Atlas 训练系列产品 ×"**（910 经典款不支持），而该样例的 OpDef 恰好注册了 `ascend910` 配置。也就是说：

- 拼装版（Maxs/Mins/Muls/Add）是**跨 SoC 可移植**的写法；
- 专用 API 版（我的选择）指令数少 4 倍，但**必须核对目标芯片在支持表里**。

这解释了为什么官方样例"故意"用低效写法——教学样例要覆盖所有 AddConfig 的芯片。教训：**API 选型 = 功能存在性 + 目标芯片支持性** 两个检查。

### 附带收获：clamp-then-scale 恒等式

`leakyrelu(x) = Maxs(x,0) + slope × Mins(x,0)` 是个通用的分段线性函数拼装套路（clamp 两半分别处理再合并），适用于没有专用指令的场景，值得记住。

## 三、代码层对比

### 我写对的（且比样例更接近生产形态）

- attr → tiling → kernel 的 slope 传递链路、OpDef 的 Attr 注册（与样例逐字一致）；
- 标量参数类型：LeakyRelu 的新原型（TensorTrait 版）要求标量 dtype 与 T 的 LiteType 一致（`enable_if is_same<PrimT<T>, U>`），我 `(T)slope` 的转换方向正确；
- CopyIn/Compute/CopyOut 三段式与样例一致（//?? 清单第 4 条销案：elementwise 就是三段式，无合并魔法）。

### 我的遗留问题（//?? 销案结果）

1. **//?? 第 2 条（DataCopyPad 参数语义）未销案**：本轮我在 host 侧把核间/片间切分全部锚定 32B 块（`everyCoreBlock`、`ubTile` 都按 32B 取整），使中间片天然对齐、plain DataCopy 安全——这算用切分设计绕开了问题。但**最后一个核的最后一块**：`totalBytesAlign` 向上取整后可能比实际 GM 数据大，plain DataCopy 有越界读风险，必须用 `DataCopyExtParams`（字节粒度 blockLen）或 `DataCopyPad` 收尾。我的代码里没有处理，是真实的 P1 边界 bug。
2. **//?? 第 5 条（realCoreNum 早退）销案**：两种模式都合法——host 把 blockDim 设成**计算后的核数**（我的做法、cann-ops add 同款）就不需要早退；host 固定 blockDim=总核数（cann-ops softmax 同款）就必须 `GetBlockIdx() >= realCoreNum` 早退。混用才会出 bug：我的 kernel 里那句 `if (blockIdx >= GetBlockNum()) return;` 是无效代码（GetBlockNum 就等于 blockDim，恒 false），应删。
3. 样例的 `loopCount = tileNum * BUFFER_NUM`（tileLength 预除 BUFFER_NUM）与我 Round 1 的错误相同——再次确认这是官方教学约定俗成的语义："每核 tileNum 个大块、每块劈 2 半做流水"。我在 Round 1 复盘已消化，本轮设计直接用了正确形式。
4. 样例 float-only、无尾块、uint32——教学样例的三件套盲区再次出现，印证 Round 1 结论："教程样例 ≠ 设计模板"。

## 四、经验教训

1. **API 选型要查产品支持矩阵**：功能存在 ≠ 目标芯片可用。可移植的拼装写法（Maxs/Mins 恒等式）是兜底手段，也是理解算子语义的透视图。
2. **切分设计可以让对齐问题消失**（全链路 32B 锚定），但**总长度的向上取整块必须用字节精确搬运收尾**（DataCopyExtParams/DataCopyPad），二者配合才完整。
3. `enable_if PrimT<T>` 模式：新 API 用类型萃取保证标量参数与张量元素类型严格一致——标量传参前显式转换成元素类型是硬要求，不是风格。
4. 无效早退代码是设计模式混淆的信号：先确定自家 host 用"计算后 blockDim"还是"固定 blockDim+kernel 早退"，kernel 侧只实现对应模式。

## 五、Round 5 计划

选 **Broadcast 类二元算子**（samples `7_broadcast_frameworklaunch`）：全新的问题类——多维索引/步长数学与 in-place 语义，检验可迁移的仍是 tiling/对齐/流水，而新挑战是逻辑维度的处理。
