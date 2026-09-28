/*
 * Round 19 - 我按文档模型写的 AllGatherMatmul（MC2 通算融合）预测式实现
 * 通信=Hccl 高阶 API（异步 handle），计算=Matmul 高阶 API；编排=本卡数据先算
 * 对没把握/外推处用 //?? 标注
 */
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "lib/hccl/hccl.h"   // //?: Hccl 高阶 API 头文件路径为外推

// ---- host/kernel 共享 tiling 结构（骨架）----
struct AllGatherMatmulTilingData {
    AscendC::Mc2InitTiling mc2InitTiling;   // //?: 字段类型待销案
    AscendC::Mc2CcTiling  mc2CcTiling;
    uint32_t tileNum;      // 主块数
    uint32_t tailNum;      // 尾块数
    uint64_t aTileEleCnt;  // 主块元素数
    uint64_t aTailEleCnt;  // 尾块元素数
    uint64_t aRankEleCnt;  // 本卡 A 全量元素数
    uint64_t aTileSize;    // 字节偏移量
    uint64_t cRankSize;    // 每 rank 的 C 尺寸（字节）
    uint64_t cTileSize;
    AscendC::tiling::TCubeTiling localTiling;   // 本卡数据形状
    AscendC::tiling::TCubeTiling tileTiling;    // 主块形状
    AscendC::tiling::TCubeTiling tailTiling;    // 尾块形状
};

using AType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using BType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using CType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using BiasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

__aicore__ inline void MatmulKernel(AscendC::GlobalTensor<half>& aGM,
                                    AscendC::GlobalTensor<half>& bGM,
                                    AscendC::GlobalTensor<float>& cGM,
                                    const AscendC::tiling::TCubeTiling& tiling,
                                    AscendC::Matmul<AType, BType, CType, BiasType>& mm)
{
    mm.Init(const_cast<AscendC::tiling::TCubeTiling*>(&tiling));
    // //?: 核间偏移与 SetTail 沿用 R6 教训——此处省略 CalcOffset，由调用方传入已偏移的 GM 视图
    mm.SetTensorA(aGM);
    mm.SetTensorB(bGM);
    mm.IterateAll(cGM);
    mm.End();
}

extern "C" __global__ __aicore__ void all_gather_matmul_custom(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                                               GM_ADDR gatherOutGM, GM_ADDR workspaceGM,
                                                               GM_ADDR tilingGM)
{
    if ASCEND_IS_AIV { return; }   // 仅 AIC 参与

    REGISTER_TILING_DEFAULT(AllGatherMatmulTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    // ---- 初始化 Hccl 高阶 API ----
    Hccl hccl;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();   // //?: 上下文获取宏为文档片段照抄
    hccl.InitV2(contextGM, &tilingData);
    hccl.SetCcTilingV2(offsetof(AllGatherMatmulTilingData, mc2CcTiling));

    // ---- 异步下发主块/尾块 AllGather ----
    auto handleId = hccl.AllGather<true>(aGM, gatherOutGM,
                                         tilingData.aTileEleCnt, HcclDataType::HCCL_DATA_TYPE_FP16,
                                         tilingData.aRankEleCnt, tilingData.tileNum);   // //?: <true>=异步外推
    auto tailHandleId = hccl.AllGather<true>(aGM + tilingData.tileNum * tilingData.aTileSize,
                                             gatherOutGM + tilingData.tileNum * tilingData.aTileSize,
                                             tilingData.aTailEleCnt, HcclDataType::HCCL_DATA_TYPE_FP16,
                                             tilingData.aRankEleCnt, tilingData.tailNum);

    // ---- 先算本卡数据（与第一轮通信互掩）----
    AscendC::TPipe pipe;
    AscendC::Matmul<AType, BType, CType, BiasType> mm;
    REGIST_MATMUL_OBJ(GetTPipePtr(), GetSysWorkSpacePtr(), mm);   // //?: MC2 场景用 GetTPipePtr 的原因待销案
    AscendC::GlobalTensor<half> aLocal, bGM, gOut;
    AscendC::GlobalTensor<float> cGMt;
    aLocal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(aGM));
    bGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bGM));
    cGMt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGM) + hccl.GetRankId() * (tilingData.cRankSize / sizeof(float)));
    MatmulKernel(aLocal, bGM, cGMt, tilingData.localTiling, mm);

    // ---- 逐轮等通信、算远端块 ----
    auto aAddr = gatherOutGM;
    auto cAddr = cGM;
    mm.Init(const_cast<AscendC::tiling::TCubeTiling*>(&tilingData.tileTiling));
    for (uint32_t i = 0; i < tilingData.tileNum; i++) {
        hccl.Wait(handleId);
        for (uint32_t rankId = 0; rankId < hccl.GetRankDim(); rankId++) {
            if (rankId == hccl.GetRankId()) continue;   // 本卡块已算过
            gOut.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(aAddr) + rankId * (tilingData.aRankEleCnt / tilingData.tileNum));
            cGMt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cAddr) + rankId * (tilingData.cRankSize / sizeof(float)));
            MatmulKernel(gOut, bGM, cGMt, tilingData.tileTiling, mm);
        }
        aAddr += tilingData.aTileSize;
        cAddr += tilingData.cTileSize;
    }

    // ---- 尾块 ----
    hccl.Wait(tailHandleId);
    mm.Init(const_cast<AscendC::tiling::TCubeTiling*>(&tilingData.tailTiling));
    // //?: 尾块 rank 循环同主块，省略

    hccl.Finalize();   // //?: Hccl 收尾接口是否存在待销案
}
