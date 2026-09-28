# Ascend C 算子开发 CHECKLIST（五十轮教训固化版）

> 用法：每轮写 DESIGN.md 前逐节过一遍，写完代码后按"代码勾销"栏逐条核对。来源见各轮复盘。

## A. 设计阶段必答（写 DESIGN.md 时逐项回答）

1. **API 选型**：基础/高阶 API 清单查过吗？高阶 API 的配套 `Get*TmpSize`/Tiling 结构找了吗？查无高阶 API 才手写。**终版（R12/R31/R48）四级半递进——①高阶 API；②库模板类(随 CANN 发布的 lib 头, 算子只做实例化分派, 如 SwigluVector)；③基础指令 API(Log/Ln/Exp/Div 等, 可组合)；④手拼多项式(仅无现成 API 的复合函数)。kernel include 的头不在仓库内=库模板类。运行时性能配置用 tiling 字段 if, 语义/精度分支用 TilingKey。**（R2/R5/R11/R12/R31/R48）
2. **产品支持矩阵**：选中的 API 在目标芯片（AddConfig 列表）支持吗？（R4）
3. **API 封装边界**：高阶 API 管到哪一层？核间映射/偏移/SetTail 是否仍是开发者的活？（R6/R7）
4. **瓶颈判定**：MTE2/V/MTE3/Cube 谁最慢？缓冲深度、流水结构据此决定。（R9）
5. **融合/物化决策**：中间结果落不落 GM？广播 y 物化还是融合消费？（R5/R7）**精化（R27）**：融合算子承担数值契约——拆开实现经过 GM 精度截断的中间量，融合版必须 round-trip cast 重建同样截断点（先量化到写出精度再继续算），保证与两算子串行逐位一致。
6. **原子 vs 两段归约**：累加宽度大→原子；标量→workspace+SyncAll+单核二次归约；要确定性→禁原子。（R10）
7. **UB 预算**：按**中间量 dtype**（fp32 常驻）算，且计入**算法临时量**（多项式逼近需 3+ 个临时张量）；多阶段用 TBufPool 取 max；单/双遍扫描据此判定。（R8/R11/R12）**精化（R22）**：方差两遍/单遍先看块是否驻留 UB——驻留→两遍（DoSub 原地，零 GM 重读，精度优先）；再权衡 UB 指令数。

## B. kernel 必检（写完代码逐条勾销）

1. **核间脚手架四件**：核间偏移（含 tailRows 折算）、SetTail、usedCoreNum 早退（或 host 计算 blockDim 模式 A）、workspace（系统+用户）。——文档伪代码不含这些！（R6/R7）
2. **DataCopy 对齐**：每一个 DataCopy 逐个检查 32B 对齐；尾块用 DataCopyPad/DataCopyExtParams 字节粒度收尾。（R2/R3/R10/R11）
3. **标量通路禁令**：GetValue/SetValue/标量循环只允许 O(核数) 级；归约结果的**所有后续计算 tensor 化**（含归一化系数的 Muls/Adds/Sqrt 批量做）。（R3/R11）
4. **2D 块意识**：tiling 原子是 UB 块不是语义单元；行算子考虑 row_factor 批行。（R5/R11）
5. **内存工具箱**：TBufPool（池中划池/第三参数复用/Reset）→ in-place（dst=src0）→ ReinterpretCast 覆盖 → 攒批写出，按粒度选用。（R8/R9/R11）
6. **L2 hint 逐 tensor**：流过数据 DISABLE 以保护复用数据，不做全局开关。（R8）
7. **BUFFER_NUM 是输出**：访存瓶颈 2 起步；块大/同步节奏快/纯物化可 1；不预设。（R2/R5/R7/R9）
8. **精度路径**：低精度输入→fp32 中间量（x² 溢出风险）；sum 累加高精度。（R2/R11）**除法精化（R20）**：标量除数→Muls(1/s)；矢量÷矢量→Div 直除（1 条胜 Reciprocal+Mul 2 条）；矢量÷标量→视除数是否复用选 Reciprocal。**精化（R32）**："低精度一律升 fp32"非绝对——有对应 half 指令且函数不敏感可留 half 域（expm1/log 的 fp16 实例化实证）；bf16 指令少普遍升 fp32；以生产为准。
9. **队列外依赖链**：临时缓冲跨多条矢量指令复用时，逐对 `PipeBarrier<PIPE_V>`；超越函数=钳位+多项式/exp 的 tensor 化逼近。（R12）
10. **归约结果读标量必须 V_S 事件对**：`SetFlag<HardEvent::V_S>` / `WaitFlag<HardEvent::V_S>` 后才能 GetValue——裸读是正确性隐患。（R13）**精化（R21）**：逐行/小归约改用掩码计数模式直读累加器——`SetMaskCount<float>() + SetVectorMask(0,rowSize) + ReduceSum(...,1) + GetAccVal()`，免 UB 往返与事件对；用完 SetMaskNorm() 恢复。**再精化（R42）**：标量读取成本决策=次数×是否 tensor 通路；逐张量低频读用 GM GetValue 直读（免 DataCopy+事件对），高频/大块才走 UB+事件对。
11. **模板特性旗标**：可选输出/精度路径用编译期模板参数（如 IS_ADDITIONAL_OUTPUT_ENABLE），不用运行时 if。（R13）
12. **Adds 双 idiom**：`Adds(-scalar)` 代减法；`Adds(dst, src, 0)` 代 UB 内拷贝（走 V 管不走 MTE）。（R13）
13. **裸流水槽复用闭环**：TBuf 槽轮转必须 MTE3_MTE2 事件对闭环（SetFlag 在 CopyOut 后、WaitFlag 在同槽 CopyIn 前）；FetchEventID 每笔依赖领一个号。（R14）
14. **缓冲×事件配对表**：无队列兜底时，DESIGN 里画缓冲清单与事件对清单的配对表，逐对自查。（R14）**加注（R24）**：反向/多输入算子是配对表事故高发区——γ 等多次使用的常驻量独立开槽，禁止复用载体（R24 现场覆写 γ 事故）。
15. **Axpy 优先**：`y += alpha*x` 场景先查 Axpy（单指令，生产 fp32 路径在用），禁手拼 Muls+Add。（R17）
16. **别名判断跳拷贝**：dst 与 src 可能同址时 `GetPhyAddr()` 比较——同址零拷贝、异址 UB→UB DataCopy 向上取整 32B（多拷尾料留 UB 无害）。（R17）
17. **算子族命名解码**：foreach 类名 arity（Binary/Ternary/Quaternary）= op 参数元数；"List" = 标量是列表（每张量一标量，ProcessPlusInLoop 逐张量 GetValue）。选基类先解码命名。（R17）
18. **GM 标量读写 tensor 化**：任何"标量直写 GM"改为 标量→S_V 事件对→写 UB→DataCopyPad(1 元素字节粒度)；V_S/S_V 成对往返（读归约结果 V_S，标量喂回 V 管 S_V）。（R18）
19. **归约 partial 是数学决策**：分段可结合的量才能两段归约；收尾算子(Sqrt/1/ord)只做一次；跨段累加全 tensor 化（段 partial 存 UB tensor 再整体 ReduceSum），标量通路只留 GetValue/SetValue 各一次。（R18）
20. **foreach 优化器算子用隐式输出**：结果原地累积进第一输入本地缓冲（Adapter 直接改写 tensor1Local，基类 needCopyOut=false），CopyOut 从 dataQueue 回队取结果写 y；省一次 UB 内搬运，y≡x1 时零拷贝。（R20）
21. **PipeBarrier 判据精化**：同管顺序指令不挂，跨缓冲重用/跨管依赖才挂；多余 barrier 也是性能损耗。Axpy 内部依赖无需外挂 barrier。（R20）
22. **权重系数隐藏加载**：γ/β 等只在 normalize 阶段需要的 GM 数据，藏进 mean/var 归约循环内用 FetchEventID+V_MTE2/MTE2_V 事件对预取，别串行在循环前。（R21）
23. **cast 双半区布局**：同一 buffer 低半区 fp32、高半区原精度（ReinterpretCast 偏移寻址），DataCopyPad 落原精度区、Cast 原地转 fp32 区。（R21）
24. **短行批归约转置**：行宽不足归约向量宽度时，考虑 UB 内转置（DoTranspose/DoReshape 类）凑归约宽度；长行场景不转置。倒数统一 Duplicate(1)+Div 张量通路，避免标量除法回程。（R22）
25. **系数隐藏加载看尺寸**：γ/β 小→块循环前一次载入常驻；大→藏进归约循环预取（B22）。适用性取决于系数张量大小。（R22 精化 B22）
26. **系数幅度 base-swap**：插值/加权算子按系数幅度分支（阈值 0.5）——代数恒等翻转基向量，把 Axpy 系数拉进最小舍入区间；数据相关分支在 kernel 内运行时做，不进 TilingKey。（R23）
27. **双输出反向算子**：逐元素通路(dx)+跨行归约通路(dgamma)并存；dgamma 用 UB 累加器（行循环 Add，循环外一次写出）；跨段和用二叉树向量 Add 折半；正向留中间量(rstd/mean)作反向输入可省反向一次归约。（R24）
28. **跨核聚合三态**：①按归约轴分核→分区即终值；②原子直写 SetAtomicAdd+DataCopyPad+SetAtomicNone（快但乱序，前置 InitOutput 清零+SyncAll 可见）；③workspace 每核槽+SyncAll+单核串行聚合（确定性）。选择=确定性需求 vs 聚合成本（小输出串行可接受）。（R25）
29. **大行数归约精度**：折半树形求和（Add self+offset 减半迭代+WholeReduceSum 收口），误差 O(log n)；常规精度够用不开启（UB 流量翻倍）。per-row 标量作用到行用 BroadCast API 张量化（tmpSize 三件套）。（R26）
30. **三输出融合缓冲计划**：主输出+残差写出+统计量攒批——残差写出用主输入槽位、γ 的 fp32 槽复用计算缓冲、rstd 独立小队列；Div(1/·) 需 sqrt 结果与全 1 向量双槽。（R27）
31. **行 max 快路径与动态量化**：Abs 独立载体→strided Max 折叠树(dstRepStride=0+WholeReduceMax 收口，禁 mask-count 慢路径)→标量 127/max→Muls(1/scale) 广播→两段 cast int8；量化输出对(int8+fp32 scale)是融合链独立截断契约点。（R28）
32. **索引访存**：过滤/压缩=CompareScalar+And+GatherMask(掩码位模式)+rsvdCnt(禁标量过滤循环)；行选(连续)逐行 DataCopy；内维离散才 GatherMask 索引模式；布局整理可 VREDUCE_MASK_ALL+repeatStride 抽行。瓶颈在访存离散度。（R33）
33. **sorted 散写聚合**：排序使重复索引连续→段内 DataCopy+Add、段尾一次写（确定性零原子）；跨核段=段首核独占前向扫描（读邻居 GM 换同步）；控制流标量化正当（B3 精化：禁的是数据计算走标量）；聚合问题可前移到数据布局（排序）。（R34）
34. **UB 常驻累加器+原子内化**：多行累加→1 行（embedding 反向/段求和）用 UB 常驻累加器行+UB 内原子加（SetAtomicAdd 族），段尾一次写出免 GM 读改写；确定性要求高换纯 Add（determinist 变体模式）；归约维超 UB→累加器沿归约维分段驻留。（R35）
35. **TopK/选择类**：proposal bit-packing(score+index 打包成可比较元素)→块内排序→MrgSort4 四路归并(ping-pong, ifExhaustedSuspension 提前暂停)→解码；pad 负无穷少量标量填充正当；CreateVecIndex 生成索引向量；最小化用取负复用。（R36）
36. **many-short-rows 批归约三路线**：跨行 repeat-Add 分块累加(Add(dst,src[col],dst,elemNum,repeat=rows,{1,1,1,blk,repStride,blk})，列块循环+行 repeat 并行) / UB 内转置 / 逐行循环；行级批量归约结果用 WholeReduceSum(mask=PLACEHOLDER,repeat=N) 一次出 N 行，天然 tensor 化。（R37）
37. **norm 族三形态终表**：驻留单位=行长 vs UB——整矩阵(single-read 单遍)/多行块(transpose 两遍)/单行列块循环(两遍)；xOut 类附加输出在首次扫描流式写出；标量累加允许级=O(chunks)且 chunks 少。（R39）
38. **量化过程化**：量化作为散写路径变换时契约=var(int8)+varScale 联合一致；by-one/by-ele 粒度是 TilingMode 一维；确定性可由索引结构保证唯一偏移（免原子免聚合）；TilingKey 位宽预算=变体维度组合数，提前规划。（R40）
39. **变长段聚合**：offsets 驱动段边界(GetValue×2)+段内索引标量驱动；MEAN 惰性除(bagSize>0)；MAX 用 −inf 初始化+argmax 簿记同步更新；簿记输出(bagSize/offset2bag)主循环增量维护禁第三遍扫描；事件对封装成具名函数(SyncM2toV 等)便于配对自查。（R43）
40. **谓词压缩+动态 shape**：CompareScalar→位图→bit-packed GatherMask(rsvdCnt=输出长度,免索引列表)；索引来自上游才用索引模式；动态输出 shape=上界预分配+字节精确写出+shape 张量回传；位图宽度随元素宽度缩放(int64 需 GM 拆读)。（R44）
41. **LSE 类(交叉熵)**：rowMax 稳定→Σexp→log；logProb=log-softmax 是 LSE 副产品顺带产出；target 标量索引驻留行随机访问；reduction(none/mean/sum)=攒批后批级聚合；label smoothing 依赖链决定扫描时机(不可流式)。（R45）
42. **全量标量输出**：ReduceSumBisect(每次折半保持 8 元素块对齐, scale 收尾一次乘)+workspace 每核 8-float 槽+SyncAll+core0 ReduceSum 聚合；mode 少且静态→子类继承叠加(每层改一环节)，多/动态→TilingKey；SyncAll 签名随架构变(310p 带 syncLocal)。（R47）
43. **量化链两端截断**：int8 in→dequant(×weightScale 通道 ×actScale token)→fp32 域→×1/quantScale(Init 预计算倒数)→int8 out；scale 三来源(常量/输入张量 Init 预计算/动态)；循环序由 scale 生命周期决定；库模板类用 .hpp。（R49）**post 阶段（R53）**：确定性聚合+格式化(rescale/cast/布局转换)拆独立 post 核(workspace 三段偏移,按输出分块并行)；NZ→ND 用 Copy+CopyRepeatParams 四步长；cast 写出对齐随 dtype(fp32=8/fp16=16 元素)。确定性聚合三级演进：同核内(R25 前态)→跨核屏障(R52)→独立阶段(R53)。
50. **自适应窗口**：窗口边界预计算到张量+重叠窗口共享载入(一次 CopyIn 喂多累加器)+逐窗口除数(MEAN 除数随窗口变)；宽窗口改前缀和差分；变体按阻塞轴分派(multi_w/split_w/split_c 式)；窗口边界静态→host 算好下发，数据依赖→核内预计算到张量。（R58）
51. **多输入单队列打包**：同 tile 消费的 N 输入装一个队列槽(一次 Alloc 搬 N 输入到槽内偏移段[0]/[a]/[2a],一次 EnQue/DeQue)，bf16 打包整槽一次 Cast；usedDb tiling 字段定缓冲深度(tileNum 随之翻倍)；尾核降级单 tile(特判集中 Init)。（R59）
44. **FlashAttention 级融合**：双 Matmul 夹 Vector 三明治(bmm1 QK^T→在线 softmax→bmm2 P·V)；跨 S2 块行状态(softmaxMax,softmaxSum,accO)+rescale；softmaxMax/Sum 落 GM 供反向；任务描述符流水(extraInfo[3] 槽解耦 Cube/Vec)；L1 复用核配对(blockIdx%2+空循环补齐)；sparse 循环范围跳全 mask 块；事件 ID 两级 Alloc(独占)/Fetch(共享)。（R51）**grad 对偶（R52）**：消费 [S1,8] softmax 状态(跨算子布局契约)+SoftmaxGradFront 高阶指令+三 matmul(bTypeTranspose 复用+MatmulCallBackFunc 定制写出)+dQ 确定性双 SyncAll 聚合。**位掩码双向（R56）**：解包=Select(src0 全 1 常量,src1RepStride=0 反复读位图,dstRepStride=8 展开)→Cast bool；压缩(R44)与解包是位掩码双向原语；位存储省 GM 带宽 8 倍付 UB+转换指令；Adapter 附件类=可选功能横切模块(hasXxx 旗标挂载)。**切分轴序（R54）**：多核切分=线性任务号按轴序分解(div/mod 链)；轴序选择=相邻任务共享哪个张量的局部性(KV→B/N2 优先, Q/sparse→S1 优先)；任务数=有效任务+流水深度-1(尾部空任务排空)。**Cube 出口 Fixpipe（R57）**：L0C→GM 直写(FixpipeParams 分形描述,绕过 UB)；反向 filter 用出口原子(SetAtomicAdd+Fixpipe)或 enSequentialWrite 顺序写做确定性；NC1HWC0 分形偏移=(nL0%stepN)·baseN·Cout+(mL0%mIter)·baseM·16；Intf/ctx 引擎=上下文结构体+静态自由函数(Config 编译期配置, conv 族专属)。
## F. 通信计算融合 MC2（R19 起生效）

1. **融合判据=依赖强弱**：画依赖图，强依赖（通信结果立即被计算使用）才融合；弱依赖走任务级并行。（R19）
2. **编排方向**：通信在计算前→本卡数据先算；计算在通信前→本卡数据最后算——把不等通信的工作塞进通信窗口。（R19）
3. **TilingData 硬规则**：Mc2InitTiling 必须是第一个成员；cfg 只存 shape 原始量，派生量 kernel 算；多形状（主块/尾块/本卡）各存一份 TCubeTiling，host 用 lambda 参数化生成。（R19）
4. **gatherOut 布局 [rank][tile]**：AllGather 按 rank 拼接，rank 是外层步长——写前先画两层顺序。（R19）
5. **卡间异步=handle 句柄**：AllGather 返回 handleId、消费点 Wait；Wait 是阻塞点尽量晚（mm.Init 先于 Wait）；收尾 mm.End()→hccl.Finalize()。（R19）

## C. host 必检

1. TilingFunc 输入三元组：硬件容量（UB/核数，查平台）+ shape + dtype 长度表（不硬编码）。（R1/R2）
2. 32B 锚点：核数上限、ubTile、大小核切分全部围绕 32B 取整。（R1）
3. 属性（attr）→ tiling 下发；标量/系数不进 kernel 常量。（R4/R7）
4. TilingKey 路由变体；TCubeTiling 等结构体用 `GetTilingData<T>()` 类型化接口；host/kernel 共享 tiling 头。（R6/R7/R11）
5. 能在 host 算的标量（avgFactor、倒数等）都在 host 算好。（R11）

## D. 流程纪律

1. DESIGN 的每个条目必须在代码里有对应物，复盘逐条勾销。（R10）
2. //?? 不确定点显式标注，对比环节逐条销案；销不了的如实记录并跨轮滚动。（全轮次）
3. 对比源优先级：cann-ops 生产代码 > samples 有代码样例 > 官方文档/论坛技巧系列；samples"待补充"空壳立即换源。（R9/R10）
4. 跨轮教训重犯时升级为 CHECKLIST 条目（本文件即由此生长）。
5. **每 15-20 轮做一次元轮**（重排/决策树化/去重，见 DECISION_FLOW.md）——写不进决策树的条目即含糊教训，重写为可判定形式。（R38）**修订（R46）**：小元轮按需触发（新骨架出现即入 DECISION_FLOW，10 分钟级）；整元轮每 15-20 轮（重排/去重/缺口审计）。

## E. 算子族/新结构类（R16 起生效）

1. **识别算子族信号**：接口名带 foreach/list 前缀、host 参数是 aclTensorList → 先找公共模板层（如 common/inc/foreach），host 走宏生成（FOREACH_OPDEF）、kernel 走"运算符函数对象作模板参数"，禁止按单算子工程从头写。（R16）
2. **框架契约优先**：张量列表地址表由 launch 机制自带（kernel 内 `GetTensorAddr` 指针走表）；任何"host 手工搬元数据到设备"的设想先质疑 launch 契约是否已解决——tiling 阶段不可写设备内存。（R16）
3. **列表类多核分配归 host**：tiling 定长数组（MAX_CORE_CONT）下发每核 [tensorStart, tensorEnd) + 首尾偏移；kernel 纯执行，不做运行时 locate。（R16）
4. **架构守卫编译期化**：`#if __CCE_AICORE__ == 220` 守卫架构相关分支，配合 TilingKey 分发 dtype。（R16）
6. **工厂选代际**：新写 foreach 优先 v2（Tiling 泛型化、TPipe 内置、Predicate 对象注入、needTempBuf 旗标）；迁移成本=换基类+函数指针改 Predicate 对象，钩子契约不变。（R29）
7. **运算符注入演进**：v1 函数指针+自由函数 Adapter → v2 Predicate 对象（成员函数+static_assert 签名校验）。（R29）
8. **工厂适用中间带**：标准读-算-写且语义简单→v2 工厂；数值复杂（分支/归约）或纯搬运（退化流水）→独立类；纯搬运=单 VECIN 队列直出（TQueBind 手工等价）+BUFFER_NUM=1。（R30）
9. **ForeachImplictOutput 签名**：\<T_in, T_out, Adapter, bufferNum, paramsCount\> 进出 dtype 分离，Adapter 工作在计算精度域(cast 边界归基类)，paramsCount=fp32 中间量槽数按需声明；一行 Adapter 即完整算子（基础指令可直接映射时）。（R31）
10. **工厂模板参数即流水配置**：bufferNum/paramsCount/needCopyOut/needTempBuf 按需实例化，同工厂类配置级复用；ProcessPlusInLoop 钩子=张量粒度标量刷新（每张量处理前调用）。（R42）

## 重犯记录（激励用）

| 教训 | 首犯 | 重犯 |
| --- | --- | --- |
| 标量通路反模式 | R3 | R11 |
| 逐行处理 vs 批行 | R2 | R11 |
| DataCopy 尾块对齐 | R2 | R10/R11 |
| 信任文档伪代码完整性 | R6 | R7 |
| 标量/系数进常量 | R4 | R21/R39(自查) |
| 猜测框架机制（未验证 launch 契约） | R16 | — |
| 缓冲生命周期破坏(γ 覆写等) | R24 | R47/R48(自查拦截) |
| API 猜测未先查支持矩阵 | R4 | R16 |
