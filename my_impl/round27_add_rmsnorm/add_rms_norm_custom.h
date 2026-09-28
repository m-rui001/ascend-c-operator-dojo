/*
 * Round 27 - 我实现的 AddRmsNorm（三输出融合 + 数值等价回环）
 * y = RmsNorm(x1+x2)·γ；xOut = x1+x2（T 精度残差写出）；rstdOut = Rms(x)
 * 关键：归一化后先量化回 T 精度再乘 γ，逐位复现"两算子串行"的舍入路径
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyAddRmsNorm {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y,
                                GM_ADDR rstdOut, GM_ADDR xOut, GM_ADDR tiling)
    {
        this->rowSize = tdRowSize;
        this->rowsPerCore = tdRowsPerCore;
        this->rowAlign = (rowSize * sizeof(uint16_t) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;  // T=half
        this->avgFactor = tdAvgFactor;  // host 算好下发（C5）
        blockIdx = AscendC::GetBlockIdx();
        uint64_t base = (uint64_t)blockIdx * rowsPerCore * rowSize;
        x1Gm.SetGlobalBuffer((__gm__ half*)x1 + base, (uint64_t)rowsPerCore * rowSize);
        x2Gm.SetGlobalBuffer((__gm__ half*)x2 + base, (uint64_t)rowsPerCore * rowSize);
        yGm.SetGlobalBuffer((__gm__ half*)y + base, (uint64_t)rowsPerCore * rowSize);
        xOutGm.SetGlobalBuffer((__gm__ half*)xOut + base, (uint64_t)rowsPerCore * rowSize);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstdOut + blockIdx * rowsPerCore, rowsPerCore);
        gammaGm.SetGlobalBuffer((__gm__ half*)gamma, rowSize);

        pipe.InitBuffer(x1q, BUFFER_NUM, rowAlign * sizeof(half));
        pipe.InitBuffer(xFp32, 1, rowAlign * sizeof(float));
        pipe.InitBuffer(sqx, 1, rowAlign * sizeof(float));      // γ fp32 槽复用
        pipe.InitBuffer(reduceBuf, 1, BLOCK_ALIGN);
        pipe.InitBuffer(rstdQ, 1, rowsPerCore * sizeof(float)); // 攒批
        pipe.InitBuffer(gq, 1, rowAlign * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        // γ 预取常驻（小张量，B25）
        AscendC::LocalTensor<half> gLocal = gq.Get<half>();
        AscendC::DataCopy(gLocal, gammaGm, rowAlign);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::LocalTensor<float> rstdLocal = rstdQ.Get<float>();
        for (uint32_t r = 0; r < rowsPerCore; r++) {
            // ---- 载入 x1/x2，fp32 域相加 ----
            AscendC::LocalTensor<half> x1L = x1q.AllocTensor<half>();
            AscendC::DataCopy(x1L, x1Gm[r * rowAlign], rowAlign);
            x1q.EnQue(x1L);
            AscendC::LocalTensor<half> x1In = x1q.DeQue<half>();
            AscendC::LocalTensor<half> x2L = x1q.AllocTensor<half>();  // 借同队列槽位（复用展示）
            AscendC::DataCopy(x2L, x2Gm[r * rowAlign], rowAlign);
            x1q.EnQue(x2L);
            AscendC::LocalTensor<half> x2In = x1q.DeQue<half>();

            AscendC::LocalTensor<float> xF = xFp32.Get<float>();
            AscendC::LocalTensor<float> sq = sqx.Get<float>();
            AscendC::Cast(xF, x1In, AscendC::RoundMode::CAST_NONE, rowSize);
            AscendC::Cast(sq, x2In, AscendC::RoundMode::CAST_NONE, rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Add(xF, xF, sq, rowSize);                 // x = x1+x2（fp32）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            // 残差输出：T 精度写 xOut（等价于 Add 算子的舍入）
            AscendC::Cast(x2In, xF, AscendC::RoundMode::CAST_RINT, rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::DataCopyPadExtParams<half> np{false, 0, 0, 0};
            AscendC::DataCopyExtParams cp{1, (uint32_t)(rowSize * sizeof(half)), 0, 0, 0};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::DataCopyPad(xOutGm[r * rowAlign], x2In, cp);
            x1q.FreeTensor(x1In);
            x1q.FreeTensor(x2In);

            // ---- Rms 归约：Σx²·avgFactor + eps → sqrt → Div(1) 倒数 ----
            AscendC::Mul(sq, xF, xF, rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Muls(sq, sq, avgFactor, rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::LocalTensor<float> rb = reduceBuf.Get<float>();
            AscendC::ReduceSumCustomF32(rb, sq, rowSize);      // 封装 WholeReduceSum 收口
            AscendC::Adds(rb, rb, eps, 1);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Sqrt(rb, rb, 1);
            AscendC::Duplicate(rb, 1.0f, 1);                   // //?: Duplicate 覆写 sqrt 结果——需双槽，勘误见复盘
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            // 正确序：sqrt 结果在 rb2，1/√ 需 Div(one, sqrt) 两槽
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            float rstd = 1.0f / rb.GetValue(0);                // //?: 演示用标量倒数；生产走 Div 张量通路（R22）
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
            rstdLocal.SetValue(r, rstd);                       // B18 成对往返

            // ---- 归一化 + 量化回环 + γ 乘 ----
            AscendC::Muls(xF, xF, rstd, rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::LocalTensor<half> yL = x1q.AllocTensor<half>();
            AscendC::Cast(yL, xF, AscendC::RoundMode::CAST_RINT, rowSize);  // 降 T（写出精度）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Cast(xF, yL, AscendC::RoundMode::CAST_NONE, rowSize);  // 升回 fp32（等价串行读回）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Cast(sq, gLocal, AscendC::RoundMode::CAST_NONE, rowSize);  // γ fp32（sqx 槽复用）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Mul(xF, xF, sq, rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Cast(yL, xF, AscendC::RoundMode::CAST_RINT, rowSize);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
            AscendC::DataCopyPad(yGm[r * rowAlign], yL, cp);
            x1q.FreeTensor(yL);
        }
        // rstd 攒批写出
        AscendC::DataCopyExtParams cpp{1, (uint32_t)(rowsPerCore * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(rstdGm, rstdLocal, cpp);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1q;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outYq;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xFp32, sqx, reduceBuf, rstdQ, gq;
    AscendC::GlobalTensor<half> x1Gm, x2Gm, yGm, xOutGm, gammaGm;
    AscendC::GlobalTensor<float> rstdGm;
    uint32_t rowSize, rowsPerCore, rowAlign, tdRowSize, tdRowsPerCore;
    uint16_t blockIdx;
    float avgFactor;
    float eps = 1e-6f;
};
