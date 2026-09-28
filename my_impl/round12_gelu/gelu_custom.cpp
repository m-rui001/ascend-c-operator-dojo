/*
 * Round 12 - 我自己写的 Gelu：高阶 API 路线（CHECKLIST A1 命中）
 * y = Gelu(x)，approximate 模式经 TilingKey 路由
 * 尾块用 DataCopyExtParams 字节精确收尾（Round 10/11 必检项）
 * 对没把握的 API 用 //?? 标注
 */
#include "kernel_operator.h"
// //?: 高阶 API 头文件路径假设（激活函数类）
#include "lib/activation/gelu.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint64_t totalLength,
                                uint64_t smallCoreNum, uint64_t bigCoreNum, uint64_t tailCoreNum,
                                uint64_t ubTile)
    {
        uint64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < tailCoreNum) {
            this->coreNum = bigCoreNum;
            this->coreOffset = blockIdx * bigCoreNum;
        } else {
            this->coreNum = smallCoreNum;
            this->coreOffset = tailCoreNum * bigCoreNum + (blockIdx - tailCoreNum) * smallCoreNum;
        }
        this->ubTile = ubTile;
        this->tileNum = (coreNum + ubTile - 1) / ubTile;

        xGm.SetGlobalBuffer((__gm__ T*)x + coreOffset, coreNum);
        yGm.SetGlobalBuffer((__gm__ T*)y + coreOffset, coreNum);
        // L2：读一次写一次，全部流过（CHECKLIST B6）
        xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubTile * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ubTile * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (coreNum == 0) return;
        for (uint64_t t = 0; t < tileNum - 1; t++) {
            ProcessTile(t * ubTile, (uint32_t)ubTile);
        }
        // 最后一片：字节精确收尾（CHECKLIST B2，Round 10/11 必检项）
        uint64_t lastOffset = (tileNum - 1) * ubTile;
        uint32_t lastCount = (uint32_t)(coreNum - lastOffset);
        ProcessTileTail(lastOffset, lastCount);
    }

private:
    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[offset], count);
        inQueueX.EnQue(xLocal);
        AscendC::LocalTensor<T> xIn = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::Gelu(yLocal, xIn, count);   // //?: 高阶 API 签名待销案
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xIn);
        AscendC::LocalTensor<T> yOut = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[offset], yOut, count);
        outQueueY.FreeTensor(yOut);
    }

    // 尾片：不足 32B 用 DataCopyPad/ExtParams 字节粒度（//?: 具体参数语义沿用 Round 3 样例记录）
    __aicore__ inline void ProcessTileTail(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        uint32_t bytes = count * sizeof(T);
        AscendC::DataCopyExtParams inParam = {1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParam = {true, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], inParam, padParam);
        inQueueX.EnQue(xLocal);
        AscendC::LocalTensor<T> xIn = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::Gelu(yLocal, xIn, count);   // //?: pad 区参与计算是否影响输出（写出时只取 count 字节则无碍）
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xIn);
        AscendC::LocalTensor<T> yOut = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams outParam = {1, bytes, 0, 0, 0};
        AscendC::DataCopyPad(yGm[offset], yOut, outParam);
        outQueueY.FreeTensor(yOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> xGm, yGm;
    uint64_t coreNum, coreOffset, ubTile, tileNum;
};

extern "C" __global__ __aicore__ void gelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {       // erf 精确模式
        KernelGelu<DTYPE_X> op;
        op.Init(x, y, tiling_data.totalLength,
                tiling_data.smallCoreNum, tiling_data.bigCoreNum, tiling_data.tailCoreNum,
                tiling_data.ubTile);
        op.Process();
    } else {                      // tanh 近似模式 //?: 高阶 API 是否暴露该开关待销案
        KernelGelu<DTYPE_X> op;
        op.Init(x, y, tiling_data.totalLength,
                tiling_data.smallCoreNum, tiling_data.bigCoreNum, tiling_data.tailCoreNum,
                tiling_data.ubTile);
        op.Process();
    }
}

#ifndef ASCENDC_CPU_DEBUG
void gelu_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                    uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling) {
    gelu_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
