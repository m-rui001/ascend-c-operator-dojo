# Round 3 复盘：ReduceSum 算子（设计 + 实现 全流程对比）

> 我的实现：`my_impl/round3_reducesum/`（DESIGN.md / reduce_sum_custom.cpp）
> 对比对象：`reference/samples/operator/ascendc/0_introduction/18_unaligned_wholereduces_frameworklaunch/WholeReduceSumCustom/`

---

## 一、设计层对比

| 设计维度 | 我的实现 | 官方 WholeReduceSum 样例 |
| --- | --- | --- |
| 归约执行方式 | **逐行逐片循环**：每行每片一次 `WholeReduceSum`，结果 GetValue(0) 后标量累加 | **一次调用归约全部行**：`WholeReduceSum<T,true>(yLocal, xLocal, cols, rows, 1, 1, srcStride)`，`repeatTimes=rows` 由硬件自动重复，结果直接连续写进 yLocal——**没有任何循环** |
| 数据搬入 | 逐片 DataCopy（行×片 双重循环） | **一次 DataCopyPad 搬入整个张量**：`DataCopyExtParams{blockCount=rows, blockLen=colBytes}` 二维块描述，行内不足 32B 自动右填充（`rpad`） |
| 行对齐 | 没处理 | **UB 内每行补齐到 32B**（`colAligned = ceil(cols*sizeof/32)*32`），因为 `srcRepStride` 的单位是 32B——对齐不是可选美化，是 repeat 寻址的前提 |
| 输出写出 | `DataCopy(yGm, yLocal, rows)`（rows*sizeof(T) 常不足 32B，越界隐患） | **DataCopyPad 按字节写出**（`{1, rows*sizeof(datatype)}`），任意长度安全 |
| 标量状态 | rowSum 标量累加 + `yLocal.SetValue(r, ...)` 逐元素写 | 不需要——repeat 机制让输出天生就是按行连续的 tensor |
| double buffer | BUFFER_NUM=2 | **BUFFER_NUM=1**：整块一次进出，无流水可言（又一次验证：BUFFER_NUM 是权衡不是信仰） |
| 数据规模假设 | 试图用跨片累加支持任意 N | 本样例假设整个张量装得下 UB；更大的规模应该**按行块 chunk**（沿行方向切，每 chunk 一次 WholeReduceSum），而不是我那样沿列切片 |

### 设计层的核心误判：我不知道 repeat 这个维度存在

矢量指令的 repeat 机制是 Ascend C 基础 API 的通用模式：**一条 API 调用 = mask（单次 256bit 宽度）× repeatTimes（重复次数）× stride（重复间步长，32B 单位）**。归约类接口的 repeat 恰好天然映射到"行"维度，所以正确的 ReduceSum 设计是"一次调用吃掉整个二维块"。我的设计停留在"一次调用 = 一片数据"的思维，把硬件循环手工化成了 C++ 循环——功能等价，指令数差 1~2 个数量级。

`mask = 256 / sizeof(datatype)` 暴露了另一个事实：矢量指令宽度是 **256bit**，所有"count"参数超过 mask 时靠 repeat 展开——这也是理解一切基础 API 签名（第 7、8 个参数）的钥匙。

## 二、代码层对比

1. **API 形态认错**：我把 `WholeReduceSum` 写成 `<T, float>(accum, xIn, count)` 三个参数的"单值归约"——真实签名是 `(dst, src, mask, repeatTimes, dstRepStride, srcRepStride)` 六参数 + 布尔模板。归约结果不是"一个标量进 accum[0]"，而是 repeatTimes 个值连续排布。
2. **`DataCopyExtParams`/`DataCopyPadExtParams` 缺失**：Round 2 学了 DataCopyPad 处理一维尾块；这里升级为二维块（blockCount × blockLen 字节粒度），一次调用描述整个张量的搬运。我还在用一维 count 循环。
3. **`CopyTiling` 手动拷贝模式**：该样例没用 `GET_TILING_DATA` 宏，而是自己从 GM 逐字拷贝 tiling 结构——说明宏是糖，本质是"tiling 在 GM 上，kernel 侧读进来"。
4. 我的 `outBuf` 大小 `rows*sizeof(T)` 未做 32B 向上取整（自己都标了 //??），正确做法：`ceil_div(bytes,32)*32`。
5. **fp16 归约精度**：样例直接以 T 归约（硬件归约通路内部精度高于标量累加假想）。我设计的"fp32 标量累加"不仅慢（每片一次 UB→标量读），精度论据也不成立——但"跨片累加需要高精度载体"的方向本身没错，只是载体应该是 fp32 tensor + Adds，不是标量。

## 三、经验教训

1. **基础 API 签名里的 repeat/stride/mask 是硬件循环的暴露**。看到任何"批量"语义的接口，先问：它能不能一次吃掉我的整个数据块？能就不要自己写循环。
2. **32B 是空间（InitBuffer/对齐填充）和时间（repStride 单位）的双重单位**。UB 内 2D 布局时"每行补齐 32B"是让 stride 寻址成立的结构性代价，换来的是零循环。
3. **归约/广播类算子的正确姿势**：DataCopyPad 二维块搬入（行补齐）→ 一次 WholeReduce 全块归约 → DataCopyPad 按字节写出。
4. 数据规模策略：装得下 UB 就单 shot；装不下**沿非归约维 chunk**，保持"归约维完整"以吃满 repeat。
5. `SetValue/GetValue` 是标量通路，是最慢的读写方式；凡是能留在 tensor 通路上的数据，不要落到标量。

## 四、三轮总览（学习曲线）

| | Round 1 Add | Round 2 Softmax | Round 3 ReduceSum |
| --- | --- | --- | --- |
| 最大的盲区 | tiling 泛化性/尾块/UB 容量 | 没查高阶 API，手写三遍扫描 | 不知道 repeat 机制，硬件循环手工化 |
| 写对的骨架 | SPMD 多核 + 三级流水 | + 大小核/尾块/TBuf/精度意识 | + API 选型先行、攒批写出意识 |
| 被纠正的关键词 | 32B 对齐、TilingData、TilingKey | DataCopyPad、高阶API、二维块、BUFFER_NUM=1 | repeat/mask/repStride、DataCopyExtParams、单 shot |

**元认知**：我的实现收敛速度在变快（P0 设计错误从"不整除就算错"→"API 选错"→"参数语义猜错"），但每一轮的盲区都出现在**文档概览没展开、而我假设了默认值**的地方——API 的真实签名、指令宽度、repeat 语义。下一步如果要 Round 4，应该先系统读一遍基础 API 的参数规范（而不是再写一个新算子），把 mask/repeat/stride 语义在所有接口族上过一遍。
