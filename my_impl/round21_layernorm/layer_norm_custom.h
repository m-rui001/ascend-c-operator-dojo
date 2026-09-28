/*
 * Round 21 - 我实现的 LayerNorm（fp32 输入，single-read 单遍方差）
 * y = (x-mean)*rstd*gamma + beta；输出 mean/rstd
 * 应用：单遍方差(E[x²]-mean²)、整块行缓存、攒批写出、γ/β 延迟加载
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;  // 整块驻留 UB，无流水需求（B7 输出）

class MyLayerNorm {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
    {
        REGISTER_TILING_DEFAULT(LnTilingData);  // //?: 占位，实际由 GET_TILING_DATA 完成
        this->rowSize = lnRowSize;
        this->nRow = lnRowsPerCore;             // 本核行数
        this->rowOffset = lnRowOffset;
        this->rowAlign = (rowSize * sizeof(float) + 31) / 32 * 32 / sizeof(float);
        this->invN = 1.0f / rowSize;

        xGm.SetGlobalBuffer((__gm__ float*)x + (uint64_t)rowOffset * rowSize, (uint64_t)nRow * rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y + (uint64_t)rowOffset * rowSize, (uint64_t)nRow * rowSize);
        meanGm.SetGlobalBuffer((__gm__ float*)mean + rowOffset, nRow);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + rowOffset, nRow);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, rowSize);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, rowSize);

        uint32_t tileLen = nRow * rowAlign;                    // 整块（含行对齐 padding）
        pipe.InitBuffer(xBuf, 1, tileLen * sizeof(float));     // x 缓存（保留 scaled 后的 x-mean）
        pipe.InitBuffer(vBuf, 1, tileLen * sizeof(float));     // 中间量：scaled x / x²
        pipe.InitBuffer(wBuf, 1, rowAlign * sizeof(float));    // γ/β/行归约载体
        pipe.InitBuffer(paramBuf, 1, 32 * ((nRow * 8 + 31) / 32));  // mean/rstd 攒批 //?: 大小取整
    }

    __aicore__ inline void Process()
    {
        // ---- 载入整块 x，行内对齐 padding 区清零语义由 pad 保证 ----
        AscendC::LocalTensor<float> xLocal = xBuf.Get<float>();
        AscendC::LocalTensor<float> vLocal = vBuf.Get<float>();
        AscendC::LocalTensor<float> wLocal = wBuf.Get<float>();
        AscendC::LocalTensor<float> pLocal = paramBuf.Get<float>();

        // 攒批搬入（2D 块：nRow 行 × rowSize 字节，行间 srcStride=0）
        AscendC::DataCopyExtParams cp{nRow, (uint32_t)(rowSize * sizeof(float)), 0,
                                      (uint16_t)((rowAlign - rowSize) * sizeof(float) / 32), 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(vLocal, xGm, cp, pad);  // //?: 先落 vLocal 再 ADD 搬 x？——布局见复盘
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // γ/β 延迟加载：先算 mean（掩盖 γ 载入）
        // ---- Pass1: 逐行 mean，同时算 x² 行和 ----
        float mVal = 0.0f, vVal = 0.0f;
        for (uint32_t r = 0; r < nRow; r++) {
            uint32_t ro = r * rowAlign;
            AscendC::WholeReduceSum<float, true>(wLocal, vLocal[ro], rowSize, 1, 1, 1, 8);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            mVal = wLocal.GetValue(0) * invN;
            pLocal.SetValue(r, mVal);                 // mean 攒批
            // 单遍方差：Σx²/N - mean² —— 从同一份缓存数据算，不重读 GM
            AscendC::Mul(vLocal[ro], vLocal[ro], vLocal[ro], rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::WholeReduceSum<float, true>(wLocal, vLocal[ro], rowSize, 1, 1, 1, 8);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            float var = wLocal.GetValue(0) * invN - mVal * mVal;
            float rstd = 1.0f / sqrt(var + eps);
            pLocal.SetValue(nRow + r, rstd);          // rstd 攒批（同一 tensor 后半区）
            // ---- Pass2 即时融合：y 行 = (x-mVal)*rstd，从 xLocal 原始值算 ----
            AscendC::Adds(xLocal[ro], xLocal[ro], -mVal, rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Muls(xLocal[ro], xLocal[ro], rstd, rowSize);
        }

        // ---- γ/β 乘加（行向量广播）----
        AscendC::DataCopy(wLocal, gammaGm, rowAlign);  // γ
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        for (uint32_t r = 0; r < nRow; r++) {
            AscendC::Mul(xLocal[r * rowAlign], xLocal[r * rowAlign], wLocal, rowSize);
        }
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::DataCopy(wLocal, betaGm, rowAlign);   // β
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        for (uint32_t r = 0; r < nRow; r++) {
            AscendC::Add(xLocal[r * rowAlign], xLocal[r * rowAlign], wLocal, rowSize);
        }

        // ---- 攒批写出 y / mean / rstd ----
        AscendC::DataCopyExtParams cpy{nRow, (uint32_t)(rowSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm, xLocal, cpy);
        AscendC::DataCopyExtParams cpp{1, (uint32_t)(nRow * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(meanGm, pLocal, cpp);
        AscendC::DataCopyPad(rstdGm, pLocal[nRow], cpp);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xBuf, vBuf, wBuf, paramBuf;
    AscendC::GlobalTensor<float> xGm, yGm, meanGm, rstdGm, gammaGm, betaGm;
    uint32_t rowSize, nRow, rowAlign, rowOffset;
    float invN;
    float eps = 1e-5f;  // //?: eps 应由 tiling 下发（B/C3）
};
