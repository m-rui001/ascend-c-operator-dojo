/*
 * Round 25 - 我实现的 dgamma 跨核两段聚合（split_d 模式 mini 版）
 * Stage1: 每核对自己的行做 dgamma 部分和 → 写 user workspace 的每核槽
 * SyncAll 全核同步
 * Stage2: 核按列段跨步认领，从 workspace 累加所有核的部分和 → 写最终 dgamma
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyDgammaTwoStage {
public:
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR dgamma,
                                GM_ADDR usrWorkspace, GM_ADDR tiling)
    {
        this->rowSize = tdRowSize;
        this->rowsPerCore = tdRowsPerCore;
        this->coreNum = tdCoreNum;
        this->rowAlign = (rowSize * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        this->invN = 1.0f / rowSize;
        blockIdx = AscendC::GetBlockIdx();

        dyGm.SetGlobalBuffer((__gm__ float*)dy + (uint64_t)blockIdx * rowsPerCore * rowSize,
                             (uint64_t)rowsPerCore * rowSize);
        xGm.SetGlobalBuffer((__gm__ float*)x + (uint64_t)blockIdx * rowsPerCore * rowSize,
                            (uint64_t)rowsPerCore * rowSize);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx * rowsPerCore, rowsPerCore);
        dgammaGm.SetGlobalBuffer((__gm__ float*)dgamma, rowSize);
        // workspace：每核一个 rowSize 槽（fp32 部分和）
        wsGm.SetGlobalBuffer((__gm__ float*)usrWorkspace + (uint64_t)blockIdx * rowSize, rowSize);

        uint32_t tileLen = rowsPerCore * rowAlign;
        pipe.InitBuffer(dyBuf, 1, rowAlign * sizeof(float));   // 行循环按行搬（列切片不需要整块）
        pipe.InitBuffer(xBuf, 1, rowAlign * sizeof(float));
        pipe.InitBuffer(dgBuf, 1, rowAlign * sizeof(float));   // 本核部分和累加器
        pipe.InitBuffer(rstdBuf, 1, BLOCK_ALIGN);
        pipe.InitBuffer(sumBuf, 1, rowAlign * sizeof(float));  // Stage2 跨核求和载体
        AscendC::LocalTensor<float> dg = dgBuf.Get<float>();
        AscendC::Duplicate(dg, 0.0f, rowAlign);
    }

    __aicore__ inline void Process()
    {
        // ---- Stage1: 本核 dgamma 部分和（行循环 × UB 累加器）----
        AscendC::LocalTensor<float> rLocal = rstdBuf.Get<float>();
        AscendC::DataCopy(rLocal, rstdGm, (rowsPerCore + 7) / 8 * 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
        AscendC::LocalTensor<float> dg = dgBuf.Get<float>();

        for (uint32_t r = 0; r < rowsPerCore; r++) {
            AscendC::LocalTensor<float> dyL = dyBuf.Get<float>();
            AscendC::LocalTensor<float> xL = xBuf.Get<float>();
            AscendC::DataCopy(dyL, dyGm[r * rowAlign], rowAlign);
            AscendC::DataCopy(xL, xGm[r * rowAlign], rowAlign);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            float rstd = rLocal.GetValue(r);
            AscendC::Muls(xL, xL, rstd, rowSize);          // xNorm
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Mul(dyL, xL, dyL, rowSize);           // dgRow = xNorm*dy
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Add(dg, dg, dyL, rowSize);            // 部分和累加
        }
        // 部分和 → 本核 workspace 槽（tensor 通路整段写出）
        AscendC::DataCopyExtParams cp{1, (uint32_t)(rowSize * sizeof(float)), 0, 0, 0};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::DataCopyPad(wsGm, dg, cp);

        // ---- 全核同步（workspace 写可见）----
        AscendC::SyncAll();

        // ---- Stage2: 按列段跨步认领，累加所有核的槽 → 最终 dgamma ----
        AscendC::LocalTensor<float> sum = sumBuf.Get<float>();
        AscendC::Duplicate(sum, 0.0f, rowAlign);
        for (uint32_t c = 0; c < coreNum; c++) {
            AscendC::DataCopy(sum, wsGm2Get(c), rowAlign);   // 逐核槽读入
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::Add(sum, sum, sum, rowSize);            // //?: 占位——真实应为 Add(sum, sum, tmp) 两 buffer 轮转
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::DataCopyPad(dgammaGm, sum, cp);
    }

private:
    __aicore__ inline AscendC::GlobalTensor<float> wsGm2Get(uint32_t c)
    {
        AscendC::GlobalTensor<float> t;
        t.SetGlobalBuffer((__gm__ float*)wsBase + (uint64_t)c * rowSize, rowSize);
        return t;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dyBuf, xBuf, dgBuf, rstdBuf, sumBuf;
    AscendC::GlobalTensor<float> dyGm, xGm, rstdGm, dgammaGm, wsGm;
    GM_ADDR wsBase;
    uint32_t rowSize, rowsPerCore, coreNum, rowAlign, tdRowSize, tdRowsPerCore, tdCoreNum;
    uint16_t blockIdx;
    float invN;
};
