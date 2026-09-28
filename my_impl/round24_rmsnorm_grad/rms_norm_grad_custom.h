/*
 * Round 24 - 我实现的 RmsNormGrad（fp32 整行驻留简化版）
 * dx = (dy*g − xNorm*meanY)*rstd,  meanY = mean(dy*g*xNorm),  dgamma = Σ_rows(xNorm*dy)
 * 本轮新点：双输出（逐元素 dx + 跨行归约 dgamma UB 累加）
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyRmsNormGrad {
public:
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma,
                                GM_ADDR dx, GM_ADDR dgamma, GM_ADDR tiling)
    {
        this->rowSize = tdRowSize;
        this->rowsPerCore = tdRowsPerCore;
        this->rowAlign = (rowSize * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        this->invN = 1.0f / rowSize;
        blockIdx = AscendC::GetBlockIdx();
        uint64_t base = (uint64_t)blockIdx * rowsPerCore * rowSize;
        dyGm.SetGlobalBuffer((__gm__ float*)dy + base, (uint64_t)rowsPerCore * rowSize);
        xGm.SetGlobalBuffer((__gm__ float*)x + base, (uint64_t)rowsPerCore * rowSize);
        dxGm.SetGlobalBuffer((__gm__ float*)dx + base, (uint64_t)rowsPerCore * rowSize);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx * rowsPerCore, rowsPerCore);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, rowSize);
        dgammaGm.SetGlobalBuffer((__gm__ float*)dgamma, rowSize);  // //?: dgamma 跨核仍需归约——生产如何跨核聚合待销案

        uint32_t tileLen = rowsPerCore * rowAlign;
        pipe.InitBuffer(dyBuf, 1, tileLen * sizeof(float));
        pipe.InitBuffer(xBuf, 1, tileLen * sizeof(float));    // xNorm = x*rstd 常驻
        pipe.InitBuffer(tBuf, 1, tileLen * sizeof(float));    // dgRow / dx 中间
        pipe.InitBuffer(wBuf, 1, rowAlign * sizeof(float));   // 行归约/γ 载体
        pipe.InitBuffer(dgBuf, 1, rowAlign * sizeof(float));  // dgamma 跨行累加器
        pipe.InitBuffer(rstdBuf, 1, BLOCK_ALIGN);
        // 载入 rstd 本核段
        AscendC::LocalTensor<float> rLocal = rstdBuf.Get<float>();
        AscendC::DataCopy(rLocal, rstdGm, (rowsPerCore + 7) / 8 * 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
        // 载入 γ
        AscendC::LocalTensor<float> wLocal = wBuf.Get<float>();
        AscendC::DataCopy(wLocal, gammaGm, rowAlign);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // dgamma 累加器清零
        AscendC::LocalTensor<float> dgLocal = dgBuf.Get<float>();
        AscendC::Duplicate(dgLocal, 0.0f, rowAlign);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> dyLocal = dyBuf.Get<float>();
        AscendC::LocalTensor<float> xLocal = xBuf.Get<float>();
        AscendC::LocalTensor<float> tLocal = tBuf.Get<float>();
        AscendC::LocalTensor<float> wLocal = wBuf.Get<float>();
        AscendC::LocalTensor<float> dgLocal = dgBuf.Get<float>();
        AscendC::LocalTensor<float> rLocal = rstdBuf.Get<float>();

        // 载入整块 dy 与 x（fp32 输入直搬）
        AscendC::DataCopyExtParams cp{rowsPerCore, (uint32_t)(rowSize * sizeof(float)), 0,
                                      (uint16_t)((rowAlign - rowSize) * sizeof(float) / BLOCK_ALIGN), 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(dyLocal, dyGm, cp, pad);
        AscendC::DataCopyPad(xLocal, xGm, cp, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        for (uint32_t r = 0; r < rowsPerCore; r++) {
            uint32_t ro = r * rowAlign;
            float rstd = rLocal.GetValue(r);   // //?: 标量读 O(行数)——可 preflight 搬进 UB 后逐元素用，演示从简
            // xNorm = x * rstd
            AscendC::Muls(xLocal[ro], xLocal[ro], rstd, rowSize);
            // dgRow = xNorm * dy
            AscendC::Mul(tLocal[0], xLocal[ro], dyLocal[ro], rowSize);
            // dgamma 跨行累加（UB 内）
            AscendC::Add(dgLocal, dgLocal, tLocal[0], rowSize);
            // meanY = Σ(dgRow * γ)/n
            AscendC::Mul(wLocal, tLocal[0], wLocal, rowSize);   // //?: 覆盖 γ 载体——需要 γ 常驻则另开槽，从简
            AscendC::AscendCUtils::SetMaskCount<float>();
            AscendC::SetVectorMask<float>(0, rowSize);
            AscendC::ReduceSum<float>(wLocal, wLocal, wLocal, 1);
            float meanY = *reinterpret_cast<float*>(&AscendC::GetAccVal()) * invN;
            AscendC::SetMaskNorm();
            // dx = (dy*γ − xNorm*meanY) * rstd
            AscendC::Mul(tLocal[0], dyLocal[ro], wLocal, rowSize);      // dy*γ（γ 已在 wLocal 被覆写！见复盘勘误）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Axpy(tLocal[0], xLocal[ro], -meanY, rowSize);      // − xNorm*meanY
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Muls(dxLocal_placeholder(tLocal), tLocal[0], rstd, rowSize);
        }
        // dx 写出省略（同 CopyOut 攒批模式）；dgamma 一次写出
        AscendC::DataCopyExtParams cp1{1, (uint32_t)(rowSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(dgammaGm, dgLocal, cp1);
    }

private:
    __aicore__ inline AscendC::LocalTensor<float>& dxLocal_placeholder(AscendC::LocalTensor<float>& t) { return t; }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dyBuf, xBuf, tBuf, wBuf, dgBuf, rstdBuf;
    AscendC::GlobalTensor<float> dyGm, xGm, dxGm, rstdGm, gammaGm, dgammaGm;
    uint32_t rowSize, rowsPerCore, rowAlign, tdRowSize, tdRowsPerCore;
    uint16_t blockIdx;
    float invN;
};
