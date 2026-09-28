/*
 * Round 9 - 流水深化版 Add：单队列装载 x|y 拼接块 + BUFFER_NUM 可配置
 * 相比 Round 1 的双队列结构：队列操作 8 次/轮 → 4 次/轮，x/y 单次同步
 * 对没把握的 API 用 //?? 标注
 */
#include "kernel_operator.h"

// BUFFER_NUM 可配置：访存瓶颈型算子 2 通常足够；加大仅解耦波动，不提升稳态吞吐
#ifndef PIPELINE_BUFFER_NUM
#define PIPELINE_BUFFER_NUM 2
#endif
constexpr int32_t BUFFER_NUM = PIPELINE_BUFFER_NUM;

class KernelAddDeepPipeline {
public:
    __aicore__ inline KernelAddDeepPipeline() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // 每轮的 tile：blockLength / tileNum / BUFFER_NUM（官方流水语义，Round 1 教训）
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 单队列：槽大小 = 2 * tileLength（x|y 拼接）
        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(half) * 2);
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> buf = inQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> xLocal = buf[0];
        AscendC::LocalTensor<half> yLocal = buf[this->tileLength];  // //?: 偏移 LocalTensor 的合法性待销案
        uint64_t offset = (uint64_t)progress * this->tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[offset], this->tileLength);
        inQueue.EnQue(buf);  // 一次 EnQue，x/y 天然同步
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> buf = inQueue.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueue.AllocTensor<half>();
        // src0/src1 同属一个 Tensor 的不重叠两段 //?: 地址重叠约束是否放行，待销案
        AscendC::Add(zLocal, buf[0], buf[this->tileLength], this->tileLength);
        outQueue.EnQue(zLocal);
        inQueue.FreeTensor(buf);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueue.DeQue<half>();
        uint64_t offset = (uint64_t)progress * this->tileLength;
        AscendC::DataCopy(zGm[offset], zLocal, this->tileLength);
        outQueue.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<half> xGm, yGm, zGm;
    uint32_t blockLength, tileNum, tileLength;
};

extern "C" __global__ __aicore__ void add_deep_pipeline(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAddDeepPipeline op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void add_deep_pipeline_do(uint32_t blockDim, void* l2ctrl, void* stream,
                          uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* workspace, uint8_t* tiling) {
    add_deep_pipeline<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif
