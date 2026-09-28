# Round 19: AllGatherMatmul（MC2 通算融合）—— 我的设计（文档驱动，CHECKLIST A 节前置）

> samples 空壳、cann-ops 无代码 → 纯文档轮（atlas_ascendc_10_10033 理论 + 10034 算子实现，代码片段在文档内）。
> 本轮性质：按文档模型实现，//?? 标注我外推的部分；对比环节=核对文档片段与我外推处。

## A. CHECKLIST 设计必答

1. **API 选型**：通信=Hccl 高阶 API（`Hccl` 对象：InitV2/SetCcTilingV2/AllGather/Wait/GetRankId/GetRankDim）；计算=Matmul 高阶 API（R6）。两个高阶 API 的拼接是本轮新结构。
2. **产品矩阵**：`AddConfig("ascendxxx")` + `this->MC2().HcclGroup("group")`——OpDef 层注册为通算融合算子；仅支持单算子 API 调用（不支持 Kernel 直调/入图）。
3. **封装边界**：Hccl 的"消息区"机制（AI Core 写消息→AI CPU 轮询）被封装；但**通信-计算的掩盖编排是开发者的**——先算本卡数据还是先通信，取决于依赖方向。
4. **瓶颈**：通信与计算互掩 → 两者都不是瓶颈，**调度顺序**是。强依赖任务才值得融合（弱依赖走任务级并行）。
5. **中间量路径**：AllGather 结果直接进 gatherOutGM（显式输出缓冲），Matmul 从那里读——通信结果不进 UB 二次搬运。
6. **切分**：通信矩阵按 M 切 tileCnt 主块 + tailCnt 尾块 → **三种形状的 Matmul Tiling**（tileM/tailM/rankM）各自算一遍多核+核内切分。
7. **核间**：仅 AIC 参与（`if ASCEND_IS_AIV { return; }`）；Matmul 多核切 M/N 不切 K。

## 1. 执行流设计（依赖方向决定编排）

```
host: 三形状 MatmulTiling × 3 + Mc2CcTilingConfig→mc2InitTiling/mc2CcTiling 入 TilingData
kernel(AIC only):
  hccl.InitV2(GetHcclContext<HCCL_GROUP_ID_0>(), tiling); SetCcTilingV2(offsetof(mc2CcTiling))
  handleId      = hccl.AllGather<true>(aGM, gatherOutGM, 主块参数)     // 异步，返回 handle
  tailHandleId  = hccl.AllGather<true>(尾块参数)
  MatmulKernel(aGM, bGM, cGM + rankId*cRankSize)                      // 先算本卡数据(与第一轮通信互掩)
  for tile in tileNum:
      hccl.Wait(handleId)                                              // 等第 tile 轮通信完成
      for rank ≠ self: MatmulKernel(gatherOut + rank偏移, ...)         // 用远端块计算
  Wait(tailHandleId); 尾块同理
```

## 2. //?? 外推清单（对比环节核对文档）

1. `hccl.AllGather<true>` 的模板参数含义（true=异步？）与 handle 生命周期（Wait 后是否自动回收）。
2. 主块/尾块两次 AllGather 调用拆分是否必要（能否一次 AllGather 全量再分段 Wait——文档按两次写，我照抄）。
3. `GetTPipePtr()` 替代 `&pipe`——REGIST_MATMUL_OBJ 在 MC2 场景用框架全局 pipe 指针的原因。
4. gatherOut 的布局：[rank][tile] 还是 [tile][rank]（文档代码按 `aAddr += aTileSize` 推进 → rank 内 tile 连续？我按文档写）。
5. mc2InitTiling 与 mc2CcTiling 在 TilingData 里的字段类型（文档用 offsetof 定位，说明是内嵌结构体成员）。
6. 通信域 group 属性的读取（host 从 attr 拿 "group" 传 Mc2CcTilingConfig）。
