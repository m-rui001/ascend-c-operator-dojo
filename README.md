# Ascend C 算子自研练习 — 项目说明

目标：只读文档概览 → 自己设计并实现算子 → 对比社区实现 → 复盘总结，循环进行。

## 目录结构

```
docs_notes/            官方文档概览抓取(入门教程等)
my_impl/
  round1_add/          第1轮: Add(DESIGN.md + kernel + host + CMake)
  round2_softmax/      第2轮: Softmax(DESIGN.md + kernel + op_host)
  round3_reducesum/    第3轮: ReduceSum(DESIGN.md + kernel)
  round4_leakyrelu/    第4轮: LeakyReLU(DESIGN.md + kernel + op_host)
  round5_broadcast/    第5轮: Broadcast 二元(DESIGN.md + kernel)
  round6_matmul/       第6轮: Matmul/Cube(DESIGN.md + kernel + op_host)
  round7_mix/          第7轮: Matmul+LeakyRelu MIX融合(DESIGN.md + kernel + op_host)
  round8_memory/       第8轮: 内存语义专题(TBufPool/复用/L2)(DESIGN.md + kernel + op_host)
  round9_pipeline/     第9轮: 流水调优深化(DESIGN.md + 单队列拼接版 Add)
  round10_atomic/      第10轮: AtomicAdd/跨核同步(DESIGN.md + AtomicSum kernel)
  round11_rmsnorm/     第11轮: RmsNorm 复合算子(DESIGN.md + kernel + op_host)
  round12_gelu/        第12轮: Gelu(CHECKLIST 首战)(DESIGN.md + kernel + op_host)
  round13_addlayernorm/ 第13轮: AddLayerNorm 融合(DESIGN.md + kernel)
  round14_hardevent/   第14轮: HardEvent 事件体系(DESIGN.md + 事件驱动版 Mul)
  round15_silu/        第15轮: Silu 高阶API路线(DESIGN.md + kernel + op_host)
  round16_foreach/     第16轮: foreach 算子族(DESIGN.md + kernel + op_host)
  round17_foreach_list/ 第17轮: foreach_add_list(Ternary层验证)(DESIGN.md + kernel)
  round18_foreach_norm/ 第18轮: foreach_norm 两段归约(DESIGN.md + kernel.h)
  round19_mc2/         第19轮: AllGatherMatmul MC2(文档驱动)(DESIGN.md + kernel)
  round20_foreach_addcdiv/ 第20轮: foreach_addcdiv_scalar 隐式输出(DESIGN+kernel)
  round21_layernorm/   第21轮: LayerNorm single-read(DESIGN + kernel.h)
  round22_ln_transpose/ 第22轮: LN transpose 块状两遍方差(DESIGN + kernel.h)
  round23_lerp/        第23轮: foreach_lerp_scalar base-swap(DESIGN + kernel.h)
  round24_rmsnorm_grad/ 第24轮: RmsNormGrad 反向导数链(DESIGN + kernel.h)
  round25_dgamma_agg/  第25轮: dgamma 跨核聚合两路销案(DESIGN + kernel.h)
  round26_high_precision/ 第26轮: high_precision 销案(diff 分析轮)
  round27_add_rmsnorm/ 第27轮: AddRmsNorm 三输出融合(DESIGN + kernel.h)
  round28_quant/       第28轮: 动态量化输出对(DESIGN + kernel.h)
  round29_unary_v2/    第29轮: foreach 工厂 v2 代际对比(DESIGN + kernel.h)
  round30_copy/        第30轮: foreach_copy 纯搬运退化流水(DESIGN + kernel.h)
  round31_log/         第31轮: foreach_log 一行 Adapter(DESIGN + kernel.cpp)
  round32_expm1/       第32轮: foreach_expm1 API组合层(DESIGN + kernel.cpp)
  round33_index_select/ 第33轮: IndexSelect/GatherMask 索引访存(DESIGN + kernel.h)
  round34_scatter/     第34轮: ScatterAddWithSorted 段聚合(DESIGN + kernel.h)
  round35_emb_grad/    第35轮: EmbeddingGrad UB累加器(DESIGN + kernel.h)
  round36_topk/        第36轮: TopKV3 proposal归并(DESIGN + kernel.h)
  round37_reduce/      第37轮: reduce_common 收拢对表(归约决策表)
  round38_meta/        第38轮: 决策流程文档(元轮, DECISION_FLOW.md)
  round39_add_layernorm/ 第39轮: AddLayerNorm 三阶段驻留行(DESIGN + kernel.h)
  round40_dqus/        第40轮: 量化散写组合(DESIGN + kernel.h)
  round41_meta/        第41轮: 决策流程增补(小元轮)
  round42_scalar_list/ 第42轮: 标量列表钩子验证(DESIGN + kernel.h)
  round43_emb_bag/     第43轮: EmbeddingBag 变长段聚合(DESIGN + kernel.h)
  round44_masked_select/ 第44轮: MaskedSelect 位图压缩(DESIGN + kernel.h)
  round45_ce_loss/     第45轮: CrossEntropyLoss LSE(DESIGN + kernel.h)
  round46_meta/        第46轮: DECISION_FLOW 对表(小元轮, loss/簿记入树)
  round47_mse/         第47轮: MSELoss 全量标量归约(DESIGN + kernel.h)
  round48_swiglu/      第48轮: SwiGLU 库模板类层级(DESIGN + kernel.h)
  round49_dequant_swiglu/ 第49轮: 量化链两端截断(DESIGN + kernel.h)
  round50_stage/       第50轮: 整元轮·阶段总结(CHECKLIST 去重+契约链入树)
  round51_flash_attention/ 第51轮: FlashAttention 架构研读(DESIGN + kernel.h)
  round52_fa_grad/     第52轮: FA Grad 状态消费侧(DESIGN + kernel.h)
  round53_fa_post/     第53轮: FA Grad Post 独立后处理(DESIGN + kernel.h)
  round54_fa_split/    第54轮: FA 变体切分轴序对比(DESIGN + axis_decomposer.h)
  round55_meta/        第55轮: FA 子树入决策流程(小元轮)
  round56_drop_mask/   第56轮: Dropout 位掩码解包(DESIGN + kernel.h)
  round57_conv_fixpipe/ 第57轮: Conv Fixpipe 出口(Intf 引擎观察)
  round58_adaptive_pool/ 第58轮: 自适应池化窗口共享(DESIGN + kernel.h)
  round59_mse_grad/    第59轮: MSE Grad 多输入单队列(DESIGN + kernel.h)
reference/cann-ops-adv/ 第51轮新增: FA 级融合算子库
CHECKLIST.md           教训固化的逐项检查单(每轮 DESIGN 前置)
DECISION_FLOW.md       七步决策流程(元轮产物, 新算子先走此流程)
reference/             社区实现(对比环节才读)
  samples/             gitee.com/ascend/samples
  cann-ops/            gitee.com/ascend/cann-ops(生产算子库)
round1_复盘_add.md     每轮复盘:设计层对比+代码层对比+教训
round2_复盘_softmax.md
round3_复盘_reducesum.md
round4_复盘_leakyrelu.md
round5_复盘_broadcast.md
round6_复盘_matmul.md
round7_复盘_mix.md
round8_复盘_memory.md
round9_复盘_pipeline.md
round10_复盘_atomic.md
round11_复盘_layernorm.md
round12_复盘_gelu.md
round13_复盘_addlayernorm.md
round14_复盘_hardevent.md
round15_复盘_silu.md
round16_复盘_foreach.md
round17_复盘_foreach_list.md
round18_复盘_foreach_norm.md
round19_复盘_mc2.md
round20_复盘_foreach_addcdiv.md
round21_复盘_layernorm.md
round22_复盘_ln_transpose.md
round23_复盘_lerp.md
round24_复盘_rmsnorm_grad.md
round25_复盘_dgamma_agg.md
round26_复盘_high_precision.md
round27_复盘_add_rmsnorm.md
round28_复盘_quant.md
round29_复盘_unary_v2.md
round30_复盘_copy.md
round31_复盘_log.md
round32_复盘_expm1.md
round33_复盘_index_select.md
round34_复盘_scatter_sorted.md
round35_复盘_emb_grad.md
round36_复盘_topk.md
round37_复盘_reduce_common.md
round38_复盘_decision_flow.md
round39_复盘_add_layernorm.md
round40_复盘_dqus.md
round41_复盘_flow_augment.md
round42_复盘_scalar_list.md
round43_复盘_emb_bag.md
round44_复盘_masked_select.md
round45_复盘_ce_loss.md
round46_复盘_flow_table.md
round47_复盘_mse.md
round48_复盘_swiglu.md
round49_复盘_dequant_swiglu.md
round50_复盘_stage50.md
round51_复盘_flash_attention.md
round52_复盘_fa_grad.md
round53_复盘_fa_post.md
round54_复盘_fa_split.md
round55_复盘_fa_tree.md
round56_复盘_drop_mask.md
round57_复盘_conv_fixpipe.md
round58_复盘_adaptive_pool.md
round59_复盘_mse_grad.md
```

## 五十九轮核心教训速览

1. **Tiling 是 host 侧运行时函数**：输入=硬件容量(UB/核数)+shape+dtype；32B 对齐是切分体系锚点；大小核+尾块是常态。
2. **先查高阶 API 再手写**：SoftMax/BroadCast 等有封装(含配套 Get*TmpSize 辅助函数)；DataCopyPad 二维块搬运是标准姿势。
3. **基础 API 的 mask/repeat/stride 是硬件循环**：单次迭代=8×32B=256B；一条调用应吃掉整个二维块；标量通路(GetValue/SetValue)是最后手段。
4. **API 选型=功能存在性+目标芯片支持性**：查产品支持矩阵；拼装恒等式(如 Maxs/Mins)是可移植兜底。
5. **UB 布局是 host 的决策**：tmpSize 进 tiling、kernel 条件 InitBuffer；融合 vs 物化是广播类算子的第一决策。
6. **问清 API 的封装边界**：Matmul 高阶 API 只封装"单核内"一切，核间映射/偏移/SetTail 仍是开发者的活；分离架构上 SetDim(矢量核数)≠SetBlockDim(Cube 核数)；生产 Matmul=变体族+host 路由。
7. **融合算子有三代写法**（耦合老式/KFC 范式/BareMix 手动同步）；融合收益=中间结果不落 GM；文档伪代码不含核间脚手架——写 kernel 前必查四件：核间偏移、SetTail、usedCoreNum 早退、workspace。
8. **内存语义**：TBufPool 池中划池+第三参数声明复用+Reset 归还，多阶段 UB 预算从"求和"变"取 max"；L2 hint 是逐 tensor 的选择性保护（绕开流过数据以保护复用数据），不是全局开关。
9. **流水调优瓶颈先行**：吞吐=max(t_MTE2,t_V,t_MTE3)，缓冲深度是瓶颈分析的输出；in-place(dst=src0) 文档允许且社区在用；纯搬运路径可 TQueBind；性能类知识在官方论坛技巧系列，samples 多为空壳。
10. **原子配批量、两段归约配标量**：SetAtomicAdd/SetAtomicNone 成对、累加"宽度"大才值得原子；确定性与原子互斥（deterministic 变体并存）；细粒度流水用事件对(PipeBarrier+set_flag/wait_flag+HardEvent)。
11. **算子族走模板工厂**：foreach 类（aclTensorList）先找公共模板层——host 一行宏(FOREACH_OPDEF)生成、kernel 把运算符函数对象作模板参数、三层架构(基类寻址/流水骨架/算子语义)；张量列表地址表由 launch 契约自带(kernel 指针走表)，多核区间 host 预算下发，架构守卫编译期化(__CCE_AICORE__)；类名 arity=op元数、List=标量列表；Axpy 代 Muls+Add、GetPhyAddr 判别名跳拷贝。
12. **列表×归约**：两段归约工厂(stage1 每核 partial 写 workspace→CrossCore 一对旗(PIPE_MTE3)→stage2 跨步认领输出)；GM 标量读写必须 tensor 化(UB+DataCopyPad 1 元素)；sqrt 只在收尾；工厂与独立类并存(骨架模式一致)。
13. **MC2 通算融合**：强依赖才融合；通信=Hccl 高阶 API(handle 异步)+计算=Matmul 高阶 API；本卡数据塞进通信窗口；Mc2InitTiling 居 TilingData 首位、多形状 TCubeTiling 并存；gatherOut 布局 [rank][tile]。
14. **隐式输出与除法选型**：foreach 优化器算子=结果原地累积进第一输入缓冲(needCopyOut=false)；矢量÷矢量用 Div 直除，倒数 idiom 只适用标量/重复除数；PipeBarrier 只挂跨缓冲重用，同管顺序不挂。
15. **LayerNorm 单遍方差**：single-read 用 E[x²]-mean² 省一遍 GM 读，fp32 中间量兜精度；逐行标量归约用 SetMaskCount+GetAccVal 直读累加器；γ/β 藏进归约循环预取；cast 用双半区 buffer。
16. **驻留块用两遍方差**：块驻留 UB 时 DoSub 原地→平方→Σ 零 GM 重读且精度更好；短行批归约可 UB 内转置凑归约宽度；倒数用 Duplicate(1)+Div 张量通路。
17. **系数幅度 base-swap**：lerp/加权类按 |w|<0.5 分支翻转插值基向量，Axpy 系数落最小舍入区间；数据相关分支留 kernel 内，shape/dtype 分支才进 TilingKey。
18. **反向算子=双归约方向**：dx 逐元素通路+dgamma 跨行 UB 累加通路并存；正向留 rstd/mean 作反向输入省一次归约；变体族按归约轴×整行/分段×精度分派。
19. **跨核聚合三态**：分区即终值 / 原子直写(InitOutput 清零+SyncAll 前置) / workspace+SyncAll+单核串行(确定性)；同一份部分和选不同出口函数即选了确定性语义。
20. **high_precision=折半树形求和**：成对求和误差 O(log n)（不是 fp64/Kahan）；per-row 系数回行用 BroadCast API 张量化；diff 两变体是销案的高效手段。
21. **融合的数值契约**：AddRmsNorm 归一化后先 cast 回 T 精度再乘 γ——拆开实现经过 GM 精度截断的中间量，融合版必须 round-trip cast 逐位复现舍入路径；三输出（y/残差 xOut/rstd 攒批）。
22. **动态量化输出对**：scale=row_max(|y|)/127 逐行标量除一次+Muls(1/scale) 广播；行 max 用 strided Max 折叠树(mask-count 慢路径禁用)；int8+fp32 scale 是 LLM 推理标准出口契约。
23. **工厂 v2 代际**：Tiling 泛型化使 elewise/reduce 共用基类；TPipe 内置；Predicate 对象注入(可 static_assert 校验)；新算子优先 v2。
24. **工厂适用中间带**：纯搬运(单队列直出)与数值特化走独立类，标准读-算-写才进工厂；30 轮里程碑：从单算子到算子族/反向/量化/MC2 的完整地图成型。
25. **超越函数三级递进**：高阶API→基础指令API(Log/Ln/Exp/...)→手拼多项式；ForeachImplictOutput 进出 dtype 分离，一行 Adapter 即完整算子。
26. **API 组合层实证**：expm1=Exp+Adds(-1) 两指令；fp16 域直算 vs 升 fp32 按函数敏感度个案决定（bf16 普遍升）；同族简单算子进入"复刻成本<设计成本"区间。
27. **索引访存双模式**：GatherMask 掩码位模式做过滤压缩(Compare+And+位图)、索引模式做内维聚集；访存连续性决定路径（连续直搬/离散 GatherMask）。
28. **排序消原子**：sorted 索引使散写退化为段内求和+段尾一次写（确定性零同步）；跨核段=段首核独占前向扫描；控制流标量化正当、数据计算 tensor 化强制。
29. **UB 累加器+原子内化**：多行累加→1 行用 UB 常驻累加行+UB 原子加，段尾一次写免 GM 读改写；归约维超 UB 沿归约维分段驻留。
30. **TopK=打包-排序-归并-解码**：score+index bit-packing 让排序带出配对索引；MrgSort4 四路归并 ping-pong；最小化取负复用。
31. **批归约三路线**：跨行 repeat-Add / UB 转置 / 逐行循环；归约决策表=D 大小→N 行数→精度敏感→结果消费形态。
32. **norm 三形态终表**：整矩阵单遍/多行块两遍/单行列块循环两遍——驻留单位决定扫描次数；附加输出首扫流式写出。
33. **量化过程化+变体扇出**：量化可以是散写路径上的过程（契约=var+scale 联合）；变体维度数=算子自由度数，TilingKey 位宽提前规划。40 轮里程碑：地图覆盖矢量/Cube/MIX/内存/流水/工厂/索引/散写/量化/MC2。

## 写代码 vs 读代码的正确配比（本练习的方法论）

- 每轮先写 DESIGN.md(含"不确定点"清单)，实现时把没把握的 API 用 `//??` 标注；
- 对比环节逐条销案：设计层(表格逐维度)+代码层(P0正确性/P1性能/P2规范)；
- 教训必须落到下一轮的设计模板里(可检验：Round 2/3 的 P0 已不再出现上一轮同类错误)。

## 参考

- 官方教程: https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0006.html
- samples 仓库: https://gitee.com/ascend/samples (operator/ascendc/)
- cann-ops: https://gitee.com/ascend/cann-ops
