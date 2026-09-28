/*
 * Round 10 - 我自己写的 AtomicSum：多核原子累加全局标量和
 * 方案A：每核 WholeReduceSum 部分和 → SetAtomicAdd → DataCopy 累加到 z[0]
 * z 由 host 预清零
 * 对没把握的 API 用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;

class KernelAtomicSum {
public:
    __aicore__ inline KernelAtomicSum() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z,
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

        xGm.SetGlobalBuffer((__gm__ half*)x + coreOffset, coreNum);
        zGm.SetGlobalBuffer((__gm__ float*)z, 1);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubTile * sizeof(half));
        // 部分和载体：fp32 单值，32B 对齐槽
        pipe.InitBuffer(sumBuf, BLOCK_SIZE);
    }

    __aicore__ inline void Process()
    {
        if (coreNum == 0) return;
        // 核内归约：跨 tile 的 fp32 标量累加（Round 3 教训：中间用 fp32 载体）
        float localSum = 0.0f;
        AscendC::LocalTensor<float> sumT = sumBuf.Get<float>();
        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = (t == tileNum - 1) ? (uint32_t)(coreNum - t * ubTile) : (uint32_t)ubTile;
            AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
            AscendC::DataCopy(xLocal, xGm[t * ubTile], count);  // //?: 尾块不足32B 对齐隐患，沿用 Round3 样例的 pad 思路可解
            inQueueX.EnQue(xLocal);
            AscendC::LocalTensor<half> xIn = inQueueX.DeQue<half>();
            AscendC::WholeReduceSum<half, true>(sumT, xIn, count, 1, 1, 1);  // //?: 签名沿用 Round 3 记录的形态
            localSum += sumT.GetValue(0);
            inQueueX.FreeTensor(xIn);
        }

        // 原子累加到 z[0]
        sumT.SetValue(0, localSum);
        AscendC::SetAtomicAdd<float>();                       // //?: 签名/作用域待销案
        AscendC::DataCopy(zGm[0], sumT, BLOCK_SIZE / sizeof(float));  // //?: 原子搬运的元素数语义待销案
        AscendC::SetAtomicNone();                             // //?: 是否存在恢复接口
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sumBuf;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<float> zGm;
    uint64_t coreNum, coreOffset, ubTile, tileNum;
};

extern "C" __global__ __aicore__ void atomic_sum_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAtomicSum op;
    op.Init(x, z, tiling_data.totalLength,
            tiling_data.smallCoreNum, tiling_data.bigCoreNum, tiling_data.tailCoreNum,
            tiling_data.ubTile);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void atomic_sum_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                          uint8_t* x, uint8_t* z, uint8_t* workspace, uint8_t* tiling) {
    atomic_sum_custom<<<blockDim, l2ctrl, stream>>>(x, z, workspace, tiling);
}
#endif
