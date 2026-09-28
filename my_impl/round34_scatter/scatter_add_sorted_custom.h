/*
 * Round 34 - 我实现的 ScatterAdd 段聚合 mini（sorted 索引，单核段扫描语义）
 * 段 [s..e) 同索引：updates 段内 DataCopy+Add 累加 → 段尾一次写 output
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyScatterAddSorted {
public:
    __aicore__ inline void Init(GM_ADDR updates, GM_ADDR indices, GM_ADDR pos, GM_ADDR output,
                                uint32_t numUpdates, uint32_t segLen, GM_ADDR tiling)
    {
        this->segLen = segLen;                       // 每个索引对应的元素段长（内维宽）
        this->numUpdates = numUpdates;
        this->segAlign = (segLen * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        upGm.SetGlobalBuffer((__gm__ float*)updates, (uint64_t)numUpdates * segLen);
        idxGm.SetGlobalBuffer((__gm__ int*)indices, numUpdates);
        posGm.SetGlobalBuffer((__gm__ int*)pos, numUpdates);
        outGm.SetGlobalBuffer((__gm__ float*)output, maxIndex * segLen);  // //?: maxIndex 由 tiling 下发（示意省略）

        pipe.InitBuffer(idxQ, 1, (numUpdates + 7) / 8 * 8 * sizeof(int));
        pipe.InitBuffer(posQ, 1, (numUpdates + 7) / 8 * 8 * sizeof(int));
        pipe.InitBuffer(accBuf, 1, segAlign * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, segAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // 索引/pos 一次载入（段边界扫描用）
        AscendC::LocalTensor<int> idxLocal = idxQ.Get<int>();
        AscendC::LocalTensor<int> posLocal = posQ.Get<int>();
        AscendC::DataCopy(idxLocal, idxGm, (numUpdates + 7) / 8 * 8);
        AscendC::DataCopy(posLocal, posGm, (numUpdates + 7) / 8 * 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        uint32_t s = 0;
        while (s < numUpdates) {
            int curIdx = idxLocal.GetValue(s);
            uint32_t e = s + 1;
            while (e < numUpdates && idxLocal.GetValue(e) == curIdx) e++;   // 段边界（数据相关标量扫描，生产同型）

            // 段内聚合：updates 行 DataCopy + Add（段首行直接当累加器初值）
            int outPos = posLocal.GetValue(s);   // 段首的 pos 即写回位置
            AscendC::DataCopy(acc, upGm[(uint64_t)s * segLen], segAlign);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            for (uint32_t k = s + 1; k < e; k++) {
                AscendC::DataCopy(tmp, upGm[(uint64_t)k * segLen], segAlign);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::Add(acc, acc, tmp, segLen);
                AscendC::PipeBarrier<AscendC::PIPE_V>();
            }
            // 段尾一次写 output[curIdx]（确定性，无原子）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::DataCopyExtParams cp{1, (uint32_t)(segLen * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(outGm[(uint64_t)curIdx * segLen], acc, cp);
            s = e;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> idxQ, posQ, accBuf, tmpBuf;
    AscendC::GlobalTensor<float> upGm, outGm;
    AscendC::GlobalTensor<int> idxGm, posGm;
    uint32_t numUpdates, segLen, segAlign, maxIndex = 1024;
};
