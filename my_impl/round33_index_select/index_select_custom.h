/*
 * Round 33 - 我实现的 IndexSelect mini：pathA 行选(逐行 DataCopy) + pathB 内维 GatherMask
 * GatherMask 是首个索引访存指令：掩码位模式(压缩)与索引模式(聚集)
 * 对没把握处用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyIndexSelect {
public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR idx, GM_ADDR out,
                                uint32_t outerLen, uint32_t dimLen,   // shape [outer, dimLen]
                                uint32_t idxNum, uint8_t axis)
    {
        this->dimLen = dimLen;
        this->idxNum = idxNum;
        this->axis = axis;
        this->dimAlign = (dimLen * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        srcGm.SetGlobalBuffer((__gm__ float*)src, (uint64_t)outerLen * dimLen);
        outGm.SetGlobalBuffer((__gm__ float*)out, (uint64_t)idxNum * dimLen);
        idxGm.SetGlobalBuffer((__gm__ int32_t*)idx, idxNum);

        // pathB：src 内维 gather 需要整批行驻留（简化：单行多次 gather）
        pipe.InitBuffer(rowBuf, 1, dimAlign * sizeof(float));
        pipe.InitBuffer(idxQ, 1, (idxNum + 7) / 8 * 8 * sizeof(int32_t));
        pipe.InitBuffer(outRowQ, 1, ((uint32_t)idxNum * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * BLOCK_ALIGN);
        pipe.InitBuffer(cmpBuf, 1, ((uint32_t)idxNum + 31) / 32 * 4);  // 位图
    }

    __aicore__ inline void Process()
    {
        if (axis == 0) {
            PathA_RowSelect();
        } else {
            PathB_InnerGather();
        }
    }

private:
    // pathA：axis=0 行选——行是连续的，逐行 DataCopy 即可（无需 GatherMask）
    __aicore__ inline void PathA_RowSelect()
    {
        AscendC::LocalTensor<int32_t> idxLocal = idxQ.Get<int32_t>();
        AscendC::DataCopy(idxLocal, idxGm, (idxNum + 7) / 8 * 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::LocalTensor<float> rowLocal = rowBuf.Get<float>();
        for (uint32_t i = 0; i < idxNum; i++) {
            int64_t srcRow = idxLocal.GetValue(i);   // //?: O(idxNum) 标量读——行号本就是标量语义，允许级
            AscendC::DataCopy(rowLocal, srcGm[(uint64_t)srcRow * dimLen], rowAlign);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::DataCopyExtParams cp{1, (uint32_t)(dimLen * sizeof(float)), 0, 0, 0};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopyPad(outGm[(uint64_t)i * dimLen], rowLocal, cp);
        }
    }

    // pathB：内维 gather——按索引从行内聚集元素（GatherMask 索引模式）
    __aicore__ inline void PathB_InnerGather()
    {
        AscendC::LocalTensor<int32_t> idxLocal = idxQ.Get<int32_t>();
        AscendC::DataCopy(idxLocal, idxGm, (idxNum + 7) / 8 * 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);

        AscendC::LocalTensor<float> rowLocal = rowBuf.Get<float>();
        AscendC::LocalTensor<float> outRow = outRowQ.Get<float>();
        // 索引有效性过滤（tensor 化）：0 <= idx < dimLen
        AscendC::LocalTensor<uint8_t> cmp0 = cmpBuf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> cmp1 = cmpBuf.Get<uint8_t>()[((idxNum + 7) / 8 * 8)];
        AscendC::CompareScalar(cmp0, idxLocal, 0.0f, AscendC::CMPMODE::GE, (idxNum + 7) / 8 * 8);
        AscendC::CompareScalar(cmp1, idxLocal, (float)dimLen, AscendC::CMPMODE::LT, (idxNum + 7) / 8 * 8);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::And(cmp0, cmp0, cmp1, idxNum);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        auto pattern = cmp0.ReinterpretCast<uint32_t>();

        uint64_t rsvdCnt = 0;
        // 掩码位模式：bit=1 的索引被压缩保留（生产 CalcIdxInRange 同型）
        AscendC::LocalTensor<int32_t> idxKept = idxQ.Get<int32_t>()[(idxNum + 7) / 8 * 8];
        GatherMask(idxKept, idxLocal, pattern, true, idxNum, {1, 1, 8, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();

        // 逐行：GatherMask 索引模式聚集（//?? 索引模式参数形态待销案）
        for (uint32_t r = 0; r < outerRows; r++) {
            AscendC::DataCopy(rowLocal, srcGm[(uint64_t)r * dimLen], rowAlign);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
            GatherMask(outRow, rowLocal, idxKept, false, rsvdCnt, {1, 1, 8, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();
            AscendC::DataCopyExtParams cp{1, (uint32_t)(rsvdCnt * sizeof(float)), 0, 0, 0};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopyPad(outGm[(uint64_t)r * rsvdCnt], outRow, cp);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowBuf, idxQ, outRowQ, cmpBuf;
    AscendC::GlobalTensor<float> srcGm, outGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    uint32_t dimLen, idxNum, outerRows, dimAlign;
    uint8_t axis;
};
