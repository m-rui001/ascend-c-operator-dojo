/*
 * Round 15 - Silu 独立算子：高阶 API 路线（独立算子 → 高阶 API，CHECKLIST A1）
 * 结构沿用 R12 模板（已过检查单的四件/对齐/L2/BUFFER_NUM）
 * 对没把握的 API 用 //?? 标注
 */
#include "kernel_operator.h"
// //?: 高阶 API 头文件路径假设
#include "lib/activation/silu.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;

template <typename T>
class KernelSilu {
public:
    __aicore__ inline KernelSilu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint64_t totalLength,
                                uint64_t smallCoreNum, uint64_t bigCoreNum, uint64_t tailCoreNum,
                                uint64_t ubTile)
    {
        uint64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < tailCoreNum) {
            coreNum = bigCoreNum; coreOffset = blockIdx * bigCoreNum;
        } else {
            coreNum = smallCoreNum; coreOffset = tailCoreNum * bigCoreNum + (blockIdx - tailCoreNum) * smallCoreNum;
        }
        this->ubTile = ubTile;
        tileNum = (coreNum + ubTile - 1) / ubTile;

        xGm.SetGlobalBuffer((__gm__ T*)x + coreOffset, coreNum);
        yGm.SetGlobalBuffer((__gm__ T*)y + coreOffset, coreNum);
        xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);   // B6
        yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubTile * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ubTile * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (coreNum == 0) return;   // B1
        for (uint64_t t = 0; t < tileNum - 1; t++) {
            ProcessTile(t * ubTile, (uint32_t)ubTile, false);
        }
        ProcessTile((tileNum - 1) * ubTile, (uint32_t)(coreNum - (tileNum - 1) * ubTile), true);  // B2
    }

private:
    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t count, bool isTail)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (isTail) {
            // 尾片：字节精确收尾（B2，R12 落实模式）
            AscendC::DataCopyExtParams inParam = {1, count * sizeof(T), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParam = {true, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[offset], inParam, padParam);
        } else {
            AscendC::DataCopy(xLocal, xGm[offset], count);
        }
        inQueueX.EnQue(xLocal);
        AscendC::LocalTensor<T> xIn = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::Silu(yLocal, xIn, count);   // //?: 高阶 API 签名待销案
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xIn);
        AscendC::LocalTensor<T> yOut = outQueueY.DeQue<T>();
        if (isTail) {
            AscendC::DataCopyExtParams outParam = {1, count * sizeof(T), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yOut, outParam);
        } else {
            AscendC::DataCopy(yGm[offset], yOut, count);
        }
        outQueueY.FreeTensor(yOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> xGm, yGm;
    uint64_t coreNum, coreOffset, ubTile, tileNum;
};

extern "C" __global__ __aicore__ void silu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSilu<DTYPE_X> op;
    op.Init(x, y, tiling_data.totalLength,
            tiling_data.smallCoreNum, tiling_data.bigCoreNum, tiling_data.tailCoreNum,
            tiling_data.ubTile);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void silu_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                    uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling) {
    silu_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
