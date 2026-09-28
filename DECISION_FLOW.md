# Ascend C 算子开发决策流程（37 轮教训的决策树化）

> 用法：新算子从上往下走；每个节点给出"判断依据 → 结论 → 依据轮次"。
> 本文件是 CHECKLIST.md 的执行序视图；CHECKLIST 按域分节，本文件按时间序串联。

## 第一步：需求与约束定位

1. **输入长什么样？**（R1/R2）
   - 任意 shape？→ tiling 必须 host 动态算，禁固定 shape 假设
   - dtype 有哪些？→ 模板 + DTYPE 宏 + TilingKey 分发；bf16 需 `__CCE_AICORE__==220` 守卫（R16）
   - 非连续/未对齐？→ DataCopyPad（R2/R3）+ 前置校验（R18）
2. **目标芯片？** → AddConfig 列表 + API 产品支持矩阵（R4）；架构相关分支编译期守卫（R16）
3. **精度契约？**（R27）
   - 有拆开的等价实现吗？→ 融合版必须 round-trip cast 逐位复现舍入路径
   - 量化输出对（int8+scale）→ 截断点即契约点（R28）

## 第二步：API 选型（三级递进 + 配套件）

1. **高阶 API**（SoftMax/Matmul/Hccl/BroadCast…）→ 找配套 `Get*TmpSize` + Tiling 结构（R2/R5/R6）
2. **基础指令 API**（Add/Exp/Log/Ln/Div/MrgSort4/CreateVecIndex…）→ 能组合绝不多项式（R31/R32）
3. **手拼多项式**（仅无现成 API 的复合函数，如 Gelu）→ tensor 化逼近（R12）
4. **问清封装边界**：高阶 API 管单核还是全算子？核间脚手架（偏移/SetTail/早退/workspace）永远自检（R6/R7）

## 第三步：结构选型

1. **算子族？**（aclTensorList/foreach 前缀）→ 找公共模板层：v2 工厂（Predicate 对象）优先（R16/R29）；纯搬运与数值特化走独立类（R30）
2. **编程模型**：
   - 纯计算 → SIMD 矢量（SPMD 三级流水）
   - 矩阵 → Cube/Matmul 高阶 API（核间映射仍是开发者的，R6）
   - 矩阵+矢量耦合 → MIX 三代写法（KFC 范式优先，R7）
   - 通信+计算强依赖 → MC2（Hccl handle 异步，R19）
3. **执行策略变体**：归约轴 × 整行/分段 × 精度 → TilingKey 路由（R24/R26）

## 第四步：Tiling 设计（host 侧）

1. **输入三元组**：硬件容量（UB/L1/核数，查平台）+ shape + dtype 长度表（C1）
2. **32B 锚点**：核数上限、ubTile、大小核、尾块全部围绕 32B 取整（C2）
3. **原子单位是 UB 块**：二维块（多行×多列）优先；行数大考虑转置凑归约宽度（R22/B24）
4. **多阶段** → TBufPool 池复用，UB 预算取各阶段 max（R8）
5. **列表结构** → 每核 [tensorStart, tensorEnd)+偏移 host 预算下发（R16）
6. **能 host 算的都 host 算**：avgFactor、倒数、每核区间（C5/R16）

## 第五步：kernel 数据流（按访存模式选骨架）

1. **元素级/连续** → 三级流水 CopyIn/Compute/CopyOut；BUFFER_NUM 按瓶颈定（R9）
2. **归约** → 决策表（R37）：
   - D 小行多 → 跨行 repeat-Add 分块累加 / 转置
   - D 大 → 分段 + 折半树形（精度敏感）/ 普通 ReduceSum
   - 块驻留 UB → 两遍方差可用（R22）
3. **索引访存** → 连续直搬；离散 GatherMask（过滤=Compare+位图压缩+rsvdCnt，R33）
4. **散写聚合** → 先问"索引可排序吗"：
   - 可排序 → 排序消原子 + 段聚合（R34）/ UB 常驻累加器+原子内化（R35）；
   - 不可排序 → 跨核聚合三态（R25）：分区即终值 / 原子直写 / workspace+单核串行；
   - 确定性可由索引结构保证唯一偏移（R40）
5. **选择/排序** → 打包-排序-MrgSort4 归并-解码（R36）
6. **量化在链上**（R28/R40）→ 出口型（int8+scale 是对外契约）vs 过程型（散写路径变换，契约=var+scale 联合）；粒度 by-one/by-ele 是 TilingMode 一维
7. **多输出与簿记**（R27/R39/R43/R45）→ 主输出流式/攒批；附加输出（xOut/bagSize/offset2bag）在首次扫描流式写出，禁第三遍扫描；依赖链明确时（label smoothing 依赖 logProb）才允许额外扫描
8. **数值契约链**（R27/R40/R49）→ 公式/流程里出现精度截断点（GM 写出/量化/低精度存储）即契约点：
   - 单点回环（融合算子 round-trip cast）/ 联合一致（var+scale）/ 两端截断+多维 scale
   - 金标准 = 拆开实现；逐位复现其舍入路径
   - scale 三来源：常量 / 输入张量（Init 预计算倒数）/ 动态输出
9. **流水调优** → 判瓶颈 → 缓冲深度 → 减队列操作 → TQueBind（R9）

### 5.x 访存模式速查（第五步判据表）

| 访存特征 | 骨架 | 依据 |
| --- | --- | --- |
| 连续读写 | 三级流水 | R1 |
| 连续读+多行累加 | UB 常驻累加器+原子内化 | R35 |
| 离散读 | GatherMask 索引模式 | R33 |
| 离散写（sorted） | 段聚合+段尾一次写 | R34 |
| 离散写（乱序） | 原子直写 | R25 |
| 归约 | R37 决策表 | R3/R21/R22/R37 |
| 选择/排序 | TopK 流水 | R36 |
| 变长段聚合(offsets 驱动) | 边界 GetValue×2+段内索引驱动；MEAN 惰性除/MAX argmax 簿记 | R43 |
| 谓词压缩 | Compare→位图→bit-packed GatherMask(rsvdCnt=动态长度) | R44 |
| LSE/损失类 | rowMax 稳定→Σexp→log；logProb 副产品；reduction 批级聚合 | R45 |
| Attention 级融合 | 双 Matmul 夹 Vector 三明治+在线 softmax+任务描述符流水；详见下方 FA 子树 | R51-R54 |

### 5.y Attention（FA）子树（R51-R54）

1. **结构**：bmm1(Q·K^T, ND/NZ 双格式) → Vec1(scale/mask/在线 softmax) → bmm2(P·V)；三阶段经 `extraInfo[3]` 任务描述符槽解耦流水（每槽打包全部派生参数）。
2. **在线 softmax**：跨 S2 块维护每行 (softmaxMax, softmaxSum, accO)；新块更新 max 时旧状态按 exp(mOld−mNew) 重缩放；softmaxMax/Sum 以 [S1,8] 布局落 GM 供反向。
3. **反向**：消费 [S1,8] 状态 + `SoftmaxGradFront` 高阶指令得 dS + 三 matmul（bTypeTranspose 复用，MatmulCallBackFunc 定制写出）；dQ 确定性聚合双 SyncAll；收尾 rescale/cast/布局拆独立 post 阶段（workspace 三段偏移 dq|dk|dv）。
4. **多核切分=轴序选择**：线性任务号按轴序 div/mod 分解——轴序决定相邻任务共享哪个张量（KV 局部→B/N2 优先；Q 局部/sparse→S1 优先）；任务数=有效任务+流水深度-1（尾部空任务排空）。
5. **L1 复用**：enableL1Reuse 时 blockIdx%2 配对共享 B 矩阵，不配对补空循环。
6. **sparse**：GetS1LoopRange 按掩码三角裁剪每核有效范围，跳过全 mask 块。

## 第六步：kernel 必检（写完勾销，CHECKLIST B 节全表）

核间四件 / DataCopy 对齐 / 标量通路（数据计算禁、控制流允，R34 精化）/ 缓冲×事件配对表（B14）/ 事件对 V_S-S_V 往返（B18）/ 权重隐藏加载（B22/B25）/ 数值回环（A5 精化）

## 第七步：验证与迭代

1. CPU 孪生调试（ASCENDC_CPU_DEBUG）先跑逻辑
2. //?? 未销项跨轮滚动；重犯升级 CHECKLIST
3. 对比源优先级：cann-ops > samples 有码样例 > 官方文档/论坛
