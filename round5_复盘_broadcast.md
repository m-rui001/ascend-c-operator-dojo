# Round 5 复盘：Broadcast 二元算子（设计 + 实现 全流程对比）

> 我的实现：`my_impl/round5_broadcast/`（DESIGN.md / broadcast_add_custom.cpp，融合式广播二元加法，三模式）
> 对比对象：`reference/samples/operator/ascendc/0_introduction/7_broadcast_frameworklaunch/BroadcastCustom/`（广播展开算子，16×1→16×3）
> 注：两边问题不完全同类——样例是"广播物化拷贝"，我做的是"融合广播计算"。对比价值在于广播的处理技术。

---

## 一、设计层对比

| 设计维度 | 我的实现 | samples 样例 |
| --- | --- | --- |
| 问题形态 | 融合：z = x + y，y 按广播语义直接参与计算，不物化全量 y | 物化：把小 shape 显式展开成目标 shape 写出 |
| 广播手段 | 三模式手写分类：行广播 y 一行常驻 UB / 列广播标量通路 Adds / 无广播逐行滚动 | **专用基础 API `BroadCast<T, dim, axis>(dst, src, dstShape, srcShape[, tmp])`**，模板参数指定维度和广播轴 |
| y 的搬运量 | 行广播 O(cols)（一次常驻复用）；列广播 O(rows)（每行 1 个标量） | 展开场景必然 O(输出量) |
| 任意轴广播 | 只支持 last-dim 家族（fold 成两轴） | dim/axis/bLength 参数化，支持非 last 轴 |
| tmp UB 缓冲 | 无 | host 用 `GetBroadCastMaxMinTmpSize` 算 max/min/mid，tmpSize 进 tiling，kernel **条件 InitBuffer** |

### //?? 清单销案结果

1. **是否有专用 Broadcast API？→ 有**，`BroadCast<T, dim, axis>`，且有配套 host 辅助函数 `GetBroadCastMaxMinTmpSize`。这确认了一个通用模式（与 SoftMax 同构）：
   **基础/高阶 API ＋ 配套 host 端 TmpSize 辅助函数 ＋ 可选 tmp UB 缓冲（大小由 host 经 tiling 下发）**。以后见到任何高阶/复杂基础 API，应同时去找它的 `Get*TmpSize` 函数。
2. **列广播用标量通路还是物化？** 我选 Adds+标量（每行 GetValue 一次）；社区技术选型是物化后纯 tensor 通路。我的方案搬运量更小，但标量通路 GetValue 是 Round 3 已知的最慢读写，且行数大时 O(rows) 次标量读可能劣于一次 BroadCast+Add（O(rows×N) 指令但全 tensor 化）。结论：**小 N 大 rows 标量赢，大 N 物化赢**——没有绝对正确答案，但设计文档里应当写下这个权衡，我当时没写。
3. **多维 fold 放 host 还是 kernel？** 样例把 dim/axis/bLength 全部由 host 算好下发，kernel 不做维度数学。我的 fold 设计（host 折两轴）方向一致，但我的 kernel 里 mode 判断和 y 视图偏移仍写死两轴，泛化性弱于样例的参数化轴。
4. **in-place 约束**：两边都没展开（Round 4 读到的"100% 重叠"约束对广播算子仍悬置，列入 Round 6 问题）。

## 二、代码层对比

### 写作过程中自己抓到的三个洞（先写 DESIGN + //?? 流程的直接收益）

初稿有三处真实设计漏洞，写实现时自己发现并修掉：
1. MODE_NONE 分支忘了把 y 搬进 UB 就引用 yBuf（逻辑洞）；
2. yBuf 只在行广播模式分配，列广播/无广播路径会用到未初始化缓冲；
3. 用 rowBase 累加器模拟 y 行推进，改成成员 rowOffset 后语义清晰。

**教训：设计文档里给每个分支画数据流（谁在 UB、谁在 GM、谁在标量），比写完再 debug 便宜得多。**

### 其余对比点

- 样例 `BUFFER_NUM=1`（第三次出现单缓冲：物化类算子 UB 被输出占满，double buffer 代价过高）；
- 样例 kernel 里 `if (TILING_KEY_IS(1)) op.Process();`——TilingKey 只有一个取值时这个分支无意义，属于样板残留，不必模仿；
- 样例 host 按 max/min/mid 三档选 tmpSize 并下发——**tiling 不只传"切多少"，还传"UB 怎么布局"**，kernel 的内存组合是 host 的决策；
- 依然是 uint32 + 教学化均分（样例通病，不再展开）。

## 三、经验教训

1. **"API 三件套"模式**：`BroadCast<dim,axis>` / `GetBroadCastMaxMinTmpSize` / kernel 条件 tmpBuffer——以后每接触一个新 API，先找它的配套 host 辅助函数和 tmp 缓冲要求。
2. **融合 vs 物化是广播算子的第一设计决策**，决策依据是广播维的位置和 y 的访问模式（常驻 / 滚动 / 标量三条通路）。
3. **UB 布局是 host 的决策**：tmpSize 进 tiling 意味着同一份 kernel 代码可以按 host 策略改变内存组合，这是"kernel 逻辑与资源策略分离"的具体形式。
4. 分支较多的算子，给每个执行分支写清"每个操作数在哪个存储层级"，能在编码前暴露大多数逻辑洞。

## 五轮总回顾（递进曲线）

| 轮 | 算子 | 新问题类 | 最大盲区（被纠正） |
| --- | --- | --- | --- |
| 1 | Add | 基础流水 | tiling 泛化/尾块/UB 容量 |
| 2 | Softmax | 跨片归约依赖 | 没查高阶 API |
| 3 | ReduceSum | 硬件归约 | repeat/mask/stride 机制 |
| 4 | LeakyReLU | attr/标量/选型 | 产品支持矩阵 |
| 5 | Broadcast | 多维语义/融合决策 | API 三件套/UB 布局是 host 决策 |

**收敛情况**：P0 级"算都算不对"的错误自 Round 2 后未再出现；本轮的主要差距已是"权衡没写进文档"和"泛化轴数不足"这类设计品味问题。练习的边际收益开始转向：下一个循环若继续，应选与已练模式正交的算子类——**Matmul/Cube 通路**（A1/A2/B1/B2/CO1/CO2 多级存储、Matmul 高阶 API、fixpipe），或先回头补 in-place/双写等内存语义。
