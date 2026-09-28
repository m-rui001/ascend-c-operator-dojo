# Round 4: LeakyReLU 算子 —— 我的设计（写代码前，未看社区实现）

## 0. 本轮前置功课（Round 3 行动项：先读 API 规范）

读完了《通用说明和约束》与《高维切分API》两篇，关键确认：

1. **单次迭代 = 8 个 datablock × 32B = 256 字节**（不是 256bit，Round 3 我在复盘中写错为"256bit 指令宽度"，此处更正）。half 单迭代最多 128 元素，float 64 元素。
2. `repeatStride=8` 表示迭代间连续（默认），`dataBlockStride=1` 表示迭代内连续；`repeatTime ≤ 255`。
3. mask 有连续模式（前 n 个元素）与逐 bit 模式两种。
4. "前 n 个数据计算"形态的接口（`count` 参数）内部自动设置 mask/repeat/stride，任意 count 都安全——**未对齐尾块的计算侧无需自己操心**，只有 DataCopy 搬运侧需要对齐处理。

## 1. API 选型（Round 2 教训）

| 候选 | 结论 |
| --- | --- |
| `LeakyRelu<T>(dst, src, scalarValue, count)` 基础 API（前 n 个数据计算） | ✅ 已确认存在（API 文档 0060），一个调用处理一片，任意 count |
| 自己用 Compare+Select 拼 `x>0 ? x : slope*x` | ❌ 无必要，两倍指令开销 |
| 高阶 LeakyReLU 封装 | 未见（待对比环节确认社区做法） |

## 2. 算子分析

- 数学：`z = x ≥ 0 ? x : negativeSlope * x`（逐元素）
- 输入 x，shape 任意（elementwise，按一维 fold），fp16/fp32；negativeSlope 是**标量属性**，host 从 attr 读出，经 tiling 下发
- 输出 z 同 shape 同 dtype

## 3. 设计（吸收前三轮全部教训）

- **核间切分**：按 32B 块数均分核数（Round 1 教训：每核至少 32B），大小核处理余数，`realCoreNum` 早退（Round 2 教训）
- **核内切分**：ubTile 由 UB 容量反推（3 份 UB：x2×double buffer + z2×double buffer → `ubSize/4/BUFFER_NUM` 向下取 32B），尾块单独循环（Round 1 教训）
- **搬运**：非对齐首尾用 `DataCopyPad`（Round 2/3 教训）；中间整块用 DataCopy
- **计算**：每片一次 `LeakyRelu(dst, src, slope, count)`，尾片 count 直接传实际数（"前 n 个"形态天然安全，本轮前置功课的收益）
- **精度**：fp16 输入直接算（slope 标量转 T）；对比环节验证社区是否用 fp32 中间精度
- **dtype**：模板 + DTYPE 宏；长度全 uint64
- **TilingKey**：暂无结构性分支，不需要（对比环节验证）

## 4. 不确定点（//?? 清单）

1. negativeSlope 在 tiling 里用 float 还是 T 存？fp16 标量在 kernel 侧怎么传给 LeakyRelu（标量参数的 dtype 要求）？
2. DataCopyPad 的 ExtParams 字段语义（blockLen 单位是字节还是元素、rightPadding 的单位）——Round 3 看过样例但没系统确认
3. 尾片若不足一个 datablock（<32B），DataCopy 搬入后会读到越界 UB 吗？是否必须 pad 到 32B
4. 社区对 elementwise 算子是否把"算"和"搬"合并成单向流水（CopyIn/Compute/CopyOut 三函数还是两函数）
