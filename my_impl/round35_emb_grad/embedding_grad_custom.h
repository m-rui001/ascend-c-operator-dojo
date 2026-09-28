/*
 * Round 35 - 我实现的 EmbeddingGrad mini：UB 常驻累加器 + 同索引 UB 累加 + 变索引写出
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyEmbeddingGrad {
public:
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR out,
                                uint32_t numUpdates, uint32_t dimLen, GM_ADDR tiling)
    {
        this->numUpdates = numUpdates;
        this->dimLen = dimLen;
        this->rowAlign = (dimLen * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        blockIdx = AscendC::GetBlockIdx();
        gradGm.SetGlobalBuffer((__gm__ float*)grad, (uint64_t)numUpdates * dimLen);
        idxGm.SetGlobalBuffer((__gm__ uint32_t*)sortIndices, numUpdates);
        outGm.SetGlobalBuffer((__gm__ float*)out, tdmNumWeights * dimLen);

        pipe.InitBuffer(gradQ, 2, rowAlign * sizeof(float));    // grad 行 double buffer
        pipe.InitBuffer(idxQ, 2, 8 * sizeof(uint32_t));          // 索引小队列
        pipe.InitBuffer(accBuf, 1, rowAlign * sizeof(float));    // 常驻累加器行
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::Duplicate(acc, 0.0f, rowAlign);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        uint32_t lastId = 0xFFFFFFFF;
        bool first = true;
        for (uint32_t i = 0; i < numUpdates; i++) {
            // 载入索引（小队列）与 grad 行
            AscendC::LocalTensor<uint32_t> idxL = idxQ.AllocTensor<uint32_t>();
            AscendC::DataCopyParams one{1, (uint16_t)sizeof(uint32_t), 0, 0};
            AscendC::DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(idxL, idxGm[i], one, pad);
            idxQ.EnQue(idxL);
            idxL = idxQ.DeQue<uint32_t>();
            uint32_t curId = idxL.GetValue(0);
            idxQ.FreeTensor(idxL);

            AscendC::LocalTensor<float> g = gradQ.AllocTensor<float>();
            AscendC::DataCopy(g, gradGm[(uint64_t)i * dimLen], rowAlign);
            gradQ.EnQue(g);
            g = gradQ.DeQue<float>();

            if (curId != lastId && !first) {
                FlushRow(acc, lastId);                   // 索引变化：写出上一段
                AscendC::Duplicate(acc, 0.0f, rowAlign); // 清零重开
            }
            // UB 内原子累加（同索引多行免 GM 读改写）
            SetAtomicAddFloat(acc, g, rowAlign);
            gradQ.FreeTensor(g);
            lastId = curId;
            first = false;
        }
        FlushRow(acc, lastId);   // 最后一段
    }

private:
    __aicore__ inline void FlushRow(AscendC::LocalTensor<float>& acc, uint32_t id)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(dimLen * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(outGm[(uint64_t)id * dimLen], acc, cp);
    }
    // UB 内原子加：SetAtomicAdd 让 V 指令以原子语义累加进 acc
    __aicore__ inline void SetAtomicAddFloat(AscendC::LocalTensor<float>& acc,
                                             AscendC::LocalTensor<float>& g, uint32_t n)
    {
        AscendC::SetAtomicAdd<AscendC::PipeConfig::UB>();  // //?: UB 原子的准确 API 形态待销案
        AscendC::Add(acc, acc, g, n);
        AscendC::SetAtomicNone();
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> gradQ, idxQ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf;
    AscendC::GlobalTensor<float> gradGm, outGm;
    AscendC::GlobalTensor<uint32_t> idxGm;
    uint32_t numUpdates, dimLen, rowAlign, tdmNumWeights = 1024;
    uint16_t blockIdx;
};
