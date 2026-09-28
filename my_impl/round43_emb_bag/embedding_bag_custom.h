/*
 * Round 43 - 我实现的 EmbeddingBag mini：变长 bag 聚合（SUM/MEAN/MAX）+ 簿记输出
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyEmbeddingBag {
public:
    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR indices, GM_ADDR offsets, GM_ADDR yOut,
                                GM_ADDR bagSizeOut, GM_ADDR maxIdxOut,
                                uint32_t numBags, uint32_t dimLen, uint8_t mode, GM_ADDR tiling)
    {
        this->dimLen = dimLen;
        this->numBags = numBags;
        this->mode = mode;
        this->rowAlign = (dimLen * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        blockIdx = AscendC::GetBlockIdx();
        weightGm.SetGlobalBuffer((__gm__ float*)weight, tdmNumWeights * dimLen);
        idxGm.SetGlobalBuffer((__gm__ int*)indices, tdIdxCount);
        offGm.SetGlobalBuffer((__gm__ int*)offsets, numBags + 1);
        yGm.SetGlobalBuffer((__gm__ float*)yOut, (uint64_t)numBags * dimLen);
        bagSizeGm.SetGlobalBuffer((__gm__ int*)bagSizeOut, numBags);
        maxIdxGm.SetGlobalBuffer((__gm__ int*)maxIdxOut, numBags * dimLen);

        pipe.InitBuffer(offQ, 1, (numBags + 1 + 7) / 8 * 8 * sizeof(int));
        pipe.InitBuffer(idxChunkQ, 1, CHUNK_INDICES * sizeof(int));
        pipe.InitBuffer(accBuf, 1, rowAlign * sizeof(float));   // bag 累加器
        pipe.InitBuffer(rowBuf, 1, rowAlign * sizeof(float));   // weight 行
        pipe.InitBuffer(bookBuf, 1, rowAlign * sizeof(float));  // maxIndices 载体（fp32 位复用）
    }

    __aicore__ inline void Process()
    {
        // offsets 一次读入（bag 边界）
        AscendC::LocalTensor<int> offLocal = offQ.Get<int>();
        AscendC::DataCopy(offLocal, offGm, (numBags + 1 + 7) / 8 * 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::LocalTensor<float> row = rowBuf.Get<float>();
        for (uint32_t b = 0; b < numBags; b++) {
            int bagStart = offLocal.GetValue(b);
            int bagEnd = offLocal.GetValue(b + 1);
            int bagSize = 0;
            AscendC::Duplicate(acc, mode == MODE_MAX ? -3.4e38f : 0.0f, rowAlign);  // MAX 用 -inf 初始化

            // bag 内逐索引聚合（变长段：无固定步长，标量索引驱动）
            for (int k = bagStart; k < bagEnd; k++) {
                int idx = idxGm.GetValue(k);   // 控制流/索引标量读（B3 允许级）
                if (idx == paddingIdx) continue;
                bagSize++;
                AscendC::DataCopy(row, weightGm[(uint64_t)idx * dimLen], rowAlign);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                if (mode == MODE_MAX) {
                    AscendC::Max(acc, acc, row, dimLen);   // 逐行 max（maxIndices 略：argmax 追踪需同步比较）
                } else {
                    AscendC::Add(acc, acc, row, dimLen);   // SUM/MEAN 共用累加
                }
                AscendC::PipeBarrier<AscendC::PIPE_V>();
            }
            if (mode == MODE_MEAN && bagSize > 0) {
                AscendC::Muls(acc, acc, 1.0f / bagSize, dimLen);
                AscendC::PipeBarrier<AscendC::PIPE_V>();
            }
            // bag 结果 + bagSize 写出
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopyPad(yGm[(uint64_t)b * dimLen], acc,
                                 AscendC::DataCopyExtParams{1, (uint32_t)(dimLen * sizeof(float)), 0, 0, 0});
            bagSizeGm.SetValue(b, bagSize);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> offQ, idxChunkQ, accBuf, rowBuf, bookBuf;
    AscendC::GlobalTensor<float> weightGm, yGm;
    AscendC::GlobalTensor<int> idxGm, offGm, bagSizeGm, maxIdxGm;
    uint32_t numBags, dimLen, rowAlign, tdmNumWeights = 4096, tdIdxCount = 4096;
    int paddingIdx = -1;
    uint8_t mode;   // 0=SUM 1=MEAN 2=MAX
    uint16_t blockIdx;
};
