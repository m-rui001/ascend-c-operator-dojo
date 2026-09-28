/*
 * Round 39 - 我实现的 AddLayerNorm mini：驻留行 + 三阶段列块循环
 * Phase0 add(+bias)→xOut 流式 | Phase1 mean | Phase2 var | Phase3 normalize
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyAddLayerNorm {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR gamma, GM_ADDR beta,
                                GM_ADDR y, GM_ADDR meanOut, GM_ADDR rstdOut, GM_ADDR tiling)
    {
        this->rowSize = tdRowSize;
        this->colChunks = tdColChunks;      // 行分段数（last_dim_per_time = 每段列宽）
        this->chunkWidth = tdChunkWidth;
        this->invN = 1.0f / rowSize;
        blockIdx = AscendC::GetBlockIdx();
        uint64_t rowBase = (uint64_t)blockIdx * rowsPerCore * rowSize;
        x1Gm.SetGlobalBuffer((__gm__ float*)x1 + rowBase, (uint64_t)rowsPerCore * rowSize);
        x2Gm.SetGlobalBuffer((__gm__ float*)x2 + rowBase, (uint64_t)rowsPerCore * rowSize);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, rowSize);      // 广播模式（IS_BIAS_BROADCAST）
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, rowSize);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y + rowBase, (uint64_t)rowsPerCore * rowSize);
        meanGm.SetGlobalBuffer((__gm__ float*)meanOut + blockIdx * rowsPerCore, rowsPerCore);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstdOut + blockIdx * rowsPerCore, rowsPerCore);

        pipe.InitBuffer(xBuf, 1, rowSize * sizeof(float));     // 驻留行（四阶段共享）
        pipe.InitBuffer(vBuf, 1, chunkWidth * sizeof(float));  // 列块工作载体
        pipe.InitBuffer(inQ, 2, chunkWidth * sizeof(float));   // x1/x2 流入
        pipe.InitBuffer(outQ, 2, chunkWidth * sizeof(float));  // y 流出
        pipe.InitBuffer(wBuf, 1, rowSize * sizeof(float));     // γ/β 常驻（小系数，B25）
        pipe.InitBuffer(paramBuf, 1, rowsPerCore * 2 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // γ/β 预取常驻（B25：小系数一次载入）
        AscendC::LocalTensor<float> wLocal = wBuf.Get<float>();
        AscendC::LocalTensor<float> gLocal = wBuf.Get<float>()[0];
        AscendC::LocalTensor<float> bLocal = wBuf.Get<float>()[rowSize];  // //?: 同 TBuf 偏移双段——示意
        AscendC::DataCopy(gLocal, gammaGm, rowSize);
        AscendC::DataCopy(bLocal, betaGm, rowSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::LocalTensor<float> pLocal = paramBuf.Get<float>();
        for (uint32_t r = 0; r < rowsPerCore; r++) {
            ProcessRow(r, pLocal[r], pLocal[rowsPerCore + r]);
        }
        // mean/rstd 攒批写出
        AscendC::DataCopyExtParams cp{1, (uint32_t)(rowsPerCore * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(meanGm, pLocal, cp);
        AscendC::DataCopyPad(rstdGm, pLocal[rowsPerCore], cp);
    }

private:
    __aicore__ inline void ProcessRow(uint32_t r, AscendC::LocalTensor<float>& meanSlot,
                                      AscendC::LocalTensor<float>& rstdSlot)
    {
        AscendC::LocalTensor<float> xF = xBuf.Get<float>();
        AscendC::LocalTensor<float> v = vBuf.Get<float>();
        AscendC::LocalTensor<float> w = wBuf.Get<float>();

        // ---- Phase0: add(+bias 广播) → 驻留 + xOut 流式 ----
        for (uint32_t c = 0; c < colChunks; c++) {
            uint32_t off = c * chunkWidth;
            AscendC::LocalTensor<float> a = inQ.AllocTensor<float>();
            AscendC::DataCopy(a, x1Gm[(uint64_t)r * rowSize + off], chunkWidth);
            inQ.EnQue(a);
            AscendC::LocalTensor<float> b = inQ.AllocTensor<float>();
            AscendC::DataCopy(b, x2Gm[(uint64_t)r * rowSize + off], chunkWidth);
            inQ.EnQue(b);
            AscendC::LocalTensor<float> ai = inQ.DeQue<float>();
            AscendC::LocalTensor<float> bi = inQ.DeQue<float>();
            AscendC::Add(xF[off], ai, bi, chunkWidth);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Add(xF[off], xF[off], w[off], chunkWidth);   // bias 广播段
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            inQ.FreeTensor(ai);
            inQ.FreeTensor(bi);
            // xOut 流式写出（Phase0 内顺带，不占额外扫描）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::DataCopyPad(xOutGm[(uint64_t)r * rowSize + off], xF[off],
                                 AscendC::DataCopyExtParams{1, (uint32_t)(chunkWidth * sizeof(float)), 0, 0, 0});
        }

        // ---- Phase1: mean（列块 Mul·ReduceSum → 标量累加，O(chunks) 允许级）----
        float ave = 0.0f;
        for (uint32_t c = 0; c < colChunks; c++) {
            uint32_t off = c * chunkWidth;
            AscendC::Muls(v, xF[off], invN, chunkWidth);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::ReduceSum<float>(v, v, v, chunkWidth);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            ave += v.GetValue(0);
        }
        meanSlot.SetValue(0, ave);

        // ---- Phase2: var（原地减 mean → 平方 → Σ·invN）----
        float var = 0.0f;
        for (uint32_t c = 0; c < colChunks; c++) {
            uint32_t off = c * chunkWidth;
            AscendC::Adds(xF[off], xF[off], -ave, chunkWidth);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Mul(v, xF[off], xF[off], chunkWidth);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Muls(v, v, invN, chunkWidth);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::ReduceSum<float>(v, v, v, chunkWidth);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
            var += v.GetValue(0);
        }
        float rstd = 1.0f / sqrt(var + eps);
        rstdSlot.SetValue(0, rstd);

        // ---- Phase3: normalize → γ → β → 流式写出 ----
        for (uint32_t c = 0; c < colChunks; c++) {
            uint32_t off = c * chunkWidth;
            AscendC::Muls(v, xF[off], rstd, chunkWidth);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Mul(v, v, w[off], chunkWidth);          // γ
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Add(v, v, w[rowSize + off], chunkWidth); // β
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::LocalTensor<float> yo = outQ.AllocTensor<float>();
            AscendC::DataCopy(yo, v, chunkWidth);
            outQ.EnQue(yo);
            AscendC::LocalTensor<float> yo2 = outQ.DeQue<float>();
            AscendC::DataCopyPad(yGm[(uint64_t)r * rowSize + off], yo2,
                                 AscendC::DataCopyExtParams{1, (uint32_t)(chunkWidth * sizeof(float)), 0, 0, 0});
            outQ.FreeTensor(yo2);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xBuf, vBuf, wBuf, paramBuf;
    AscendC::GlobalTensor<float> x1Gm, x2Gm, biasGm, gammaGm, betaGm, yGm, meanGm, rstdGm, xOutGm;
    uint32_t rowSize, rowsPerCore = 4, colChunks, chunkWidth;
    uint16_t blockIdx;
    float invN;
    float eps = 1e-5f;  // //?: 重犯自查——tiling 下发（R21/R22 已记录）
};
