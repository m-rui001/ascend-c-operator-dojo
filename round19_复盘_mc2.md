# Round 19 复盘：AllGatherMatmul（MC2 通算融合，文档驱动轮）

> 我的实现：`my_impl/round19_mc2/`（DESIGN.md / all_gather_matmul_custom.cpp，预测式）
> 对比对象：官方文档 `atlas_ascendc_10_10033`（理论）+ `10_10034`（算子实现，含完整代码片段）；samples MC2 与 cann-ops 均无代码。
> 本轮性质：文档驱动轮——对比 = 核对我的外推 //?? 与文档片段。

## 一、理论层收获（10033）

1. **融合判据是依赖强弱**：弱依赖（通信可被无依赖计算掩盖）→ 不融合，走任务级并行；**强依赖**（通信结果立即被计算使用）→ 才值得 MC2 融合。选型先画依赖图。
2. **收益来源**：数据切分后通信与计算分多次交替，两者流水互掩，理论耗时从"串行加和"降为"max(通信,计算)+少量气泡"。

## 二、编程模型（10034）与我的外推核对

| //?? | 我的外推 | 文档答案 |
| --- | --- | --- |
| TilingData 字段组织 | 内嵌 mc2InitTiling/mc2CcTiling + 三个 TCubeTiling | ✅ 且新增硬规则：**Mc2InitTiling 必须是 TilingData 第一个成员**；自定义 cfg 结构放最后 |
| cfg 字段 | 我写了 eleCnt/字节尺寸等派生量 | ❌ 文档只存 shape 原始量（rankM/N/K、tileNum、tailM、tailNum），元素数/字节数在 kernel 从 tiling 派生——**tiling 存决策参数，kernel 算派生量** |
| gatherOut 布局 | 我推 [tile][rank]（aAddr 按 tile 推进） | ❌ **[rank][tile]**：`aAddr + rankId * aRankSize`（rank 是外层步长，tile 在 rank 区间内推进）。AllGather 结果按 rank 拼接，tile 是 rank 区间内的分块——我把拼接的两层顺序搞反了 |
| `hccl.Finalize()` | 猜存在 | ✅ 存在，且 `mm.End()` 在其前 |
| 尾块 Wait/Init 次序 | 我写了 Wait→Init | ❌ 文档是 `mm.Init(&tailTiling)` **先**、`hccl.Wait` 后——Wait 是阻塞点应尽量晚 |
| `<true>` 模板参数 | 猜"异步" | 未明说，但 handle+Wait 模式确证异步语义 |
| GetTPipePtr | 疑惑 | 文档代码原样使用（MC2 场景 REGIST_MATMUL_OBJ 用框架 pipe 指针） |

其余确认：host 用 lambda `matmulTilingFunc(m,n,k,tcubeTiling)` 复用三形状 tiling；`SetBufferSpace(L1_BUFFER_SIZE)` 指定 L1 预算；`SetDim(aicCoreNum)`；kernel 侧 `if ASCEND_IS_AIV { return; }` + `GetHcclContext<HCCL_GROUP_ID_0>()` + `SetCcTilingV2(offsetof(...))`。

## 三、结构层结论（MC2 的通用骨架）

1. **编排方向由依赖决定**：通信在计算前（AllGatherMatmul）→ 本卡数据先算（与第一轮通信互掩，且省掉对本卡数据的分块重复计算）；计算在通信前（MatmulAllReduce）→ 本卡数据最后算（与最后一轮通信互掩）。**先算谁 = 把"不需要等通信"的那份工作塞进通信窗口**。
2. **多形状 tiling 是常态**：通信切分产生 tileM/tailM/rankM 三种形状 → TCubeTiling 各存一份（TCubeTiling 单形状限制），host 用 lambda 参数化生成。
3. **异步通信 = handle 句柄**：AllGather 返回 handleId，消费点 `hccl.Wait(handleId)`——与 R14 事件对同构，只是同步域从核内扩到卡间。
4. **消息区抽象**：AI Core 写消息→AI CPU 轮询执行通信，被 Hccl API 封装——开发者只面对 handle。

## 四、CHECKLIST 增量（F 节新增）

## 下一轮候选

foreach_addcdiv_scalar（除法+倒数 idiom 列表版）或 foreach_lerp_scalar（三运算符插值）——低成本工厂复用验证第二弹。
