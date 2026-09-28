/*
 * Round 4 - 我自己写的 LeakyReLU 算子（elementwise，一维 fold）
 * 本轮重点：API 选型先行(基础API LeakyRelu)、repeat/stride 语义前置功课、DataCopyPad 对齐搬运
 * 对 API 细节没把握处用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;

template <typename T>
class KernelLeakyRelu {
public:
    __aicore__ inline KernelLeakyRelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z,
                                uint64_t totalLength,
                                uint64_t smallCoreNum,   // 小核元素数
                                uint64_t bigCoreNum,     // 大核元素数(前 tailCoreNum 个核)
                                uint64_t tailCoreNum,    // 大核个数
                                uint64_t ubTile,         // 每片元素数(UB 反推, 32B 对齐)
                                float negativeSlope)
    {
        uint64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= AscendC::GetBlockNum()) return;  //?? realCoreNum 早退：blockNum 是否可能 > 有效核数
        if (blockIdx < tailCoreNum) {
            this->coreNum = bigCoreNum;
            this->coreOffset = blockIdx * bigCoreNum;
        } else {
            this->coreNum = smallCoreNum;
            this->coreOffset = tailCoreNum * bigCoreNum + (blockIdx - tailCoreNum) * smallCoreNum;
        }
        this->ubTile = ubTile;
        this->slope = negativeSlope;
        this->tileNum = (coreNum + ubTile - 1) / ubTile;
        this->tailCount = coreNum % ubTile == 0 ? ubTile : coreNum % ubTile;

        xGm.SetGlobalBuffer((__gm__ T*)x + coreOffset, coreNum);
        zGm.SetGlobalBuffer((__gm__ T*)z + coreOffset, coreNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubTile * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, ubTile * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (coreNum == 0) return;  // 空核早退
        for (uint64_t t = 0; t < tileNum - 1; t++) {
            ProcessTile(t * ubTile, ubTile);
        }
        ProcessTile((tileNum - 1) * ubTile, tailCount);  // 尾片单独处理
    }

private:
    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t count) {
        // 搬入：//?? 尾片 count*sizeof(T) 不足 32B 时，DataCopy 是否安全？
        // 本设计假设 totalLength 按 32B 对齐均分核(见 host)，故中间片安全；
        // 尾片仍存疑，暂用 DataCopy（对比环节销案）
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[offset], count);
        inQueueX.EnQue(xLocal);
        AscendC::LocalTensor<T> xIn = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        // 前n个数据计算形态：内部自动 mask/repeat，任意 count 安全
        AscendC::LeakyRelu<T>(zLocal, xIn, (T)slope, count);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xIn);
        AscendC::LocalTensor<T> zOut = outQueueZ.DeQue<T>();
        AscendC::DataCopy(zGm[offset], zOut, count);
        outQueueZ.FreeTensor(zOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<T> xGm, zGm;
    uint64_t coreNum, coreOffset, ubTile, tileNum, tailCount;
    float slope;
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyRelu<DTYPE_X> op;
    op.Init(x, z, tiling_data.totalLength,
            tiling_data.smallCoreNum, tiling_data.bigCoreNum, tiling_data.tailCoreNum,
            tiling_data.ubTile, tiling_data.negativeSlope);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void leaky_relu_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                          uint8_t* x, uint8_t* z, uint8_t* workspace, uint8_t* tiling) {
    leaky_relu_custom<<<blockDim, l2ctrl, stream>>>(x, z, workspace, tiling);
}
#endif
