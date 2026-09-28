/*
 * Round 40 - 我实现的 DynamicQuantUpdateScatter mini：by-one 量化 + 按索引散写
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;
constexpr float QUANT_DIVIDEND = 127.0f;

class MyDQUS {
public:
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR updates,
                                uint32_t numUpdates, uint32_t dimLen, GM_ADDR tiling)
    {
        this->dimLen = dimLen;
        this->rowAlign = (dimLen * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        blockIdx = AscendC::GetBlockIdx();
        upGm.SetGlobalBuffer((__gm__ float*)updates, (uint64_t)numUpdates * dimLen);
        idxGm.SetGlobalBuffer((__gm__ int*)indices, numUpdates);
        varGm.SetGlobalBuffer((__gm__ int8_t*)var, tdmVarRows * dimLen);
        scaleGm.SetGlobalBuffer((__gm__ float*)varScale, tdmVarRows);

        pipe.InitBuffer(upQ, 2, rowAlign * sizeof(float));
        pipe.InitBuffer(idxQ, 2, 8 * sizeof(int));
        pipe.InitBuffer(vBuf, 1, rowAlign * sizeof(float));
        pipe.InitBuffer(i8Buf, 1, rowAlign);
        // 散写目标不预清零——每个 (idx) 若被多次更新需原子或排序段聚合（R34 前置条件；//?: 语义按调用方保证唯一）
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < numUpdates; i++) {
            // 索引读入（标量语义）
            AscendC::LocalTensor<int> idxL = idxQ.AllocTensor<int>();
            AscendC::DataCopyParams one{1, (uint16_t)sizeof(int), 0, 0};
            AscendC::DataCopyPadExtParams<int> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(idxL, idxGm[i], one, pad);
            idxQ.EnQue(idxL);
            idxL = idxQ.DeQue<int>();
            int dstIdx = idxL.GetValue(0);
            idxQ.FreeTensor(idxL);

            // updates 行载入 → Abs → 行 max → scale → 量化
            AscendC::LocalTensor<float> up = upQ.AllocTensor<float>();
            AscendC::DataCopy(up, upGm[(uint64_t)i * dimLen], rowAlign);
            upQ.EnQue(up);
            up = upQ.DeQue<float>();
            AscendC::LocalTensor<float> v = vBuf.Get<float>();
            AscendC::Abs(v, up, dimLen);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::WholeReduceMax<float, false>(v, v, dimLen < 64 ? dimLen : 64, 1, 8, 1, 8);  // //?: 大行需折叠树（R28 B31）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            float maxV = v.GetValue(0);
            float scale = QUANT_DIVIDEND / maxV;

            // 广播除法 → int8 写 var[idx]；scale 写 varScaleOut[idx]
            AscendC::Muls(up, up, 1.0f / scale, dimLen);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::LocalTensor<half> hTmp = vBuf.Get<half>();
            AscendC::Cast(hTmp, up, AscendC::RoundMode::CAST_RINT, dimLen);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::LocalTensor<int8_t> i8 = i8Buf.Get<int8_t>();
            AscendC::Cast(i8, hTmp, AscendC::RoundMode::CAST_RINT, dimLen);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::DataCopyExtParams cp{1, (uint32_t)dimLen, 0, 0, 0};
            AscendC::DataCopyPad(varGm[(uint64_t)dstIdx * dimLen], i8, cp);
            AscendC::DataCopyExtParams cs{1, (uint32_t)sizeof(float), 0, 0, 0};
            AscendC::DataCopyPad(scaleGm[dstIdx], v, cs);  // scale 存回（//?: 载体混用示意）
            upQ.FreeTensor(up);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> upQ, idxQ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> vBuf, i8Buf;
    AscendC::GlobalTensor<float> upGm, scaleGm;
    AscendC::GlobalTensor<int> idxGm;
    AscendC::GlobalTensor<int8_t> varGm;
    uint32_t numUpdates, dimLen, rowAlign, tdmVarRows = 1024;
    uint16_t blockIdx;
};
