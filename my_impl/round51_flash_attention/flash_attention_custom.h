/*
 * Round 51 - 我实现的 FlashAttention 骨架 mini（单核、S2 循环在线 softmax）
 * Q[S1,D] K[S2,D] V[S2,D] → O[S1,D]；行状态 (softmaxMax, softmaxSum, accO) 跨块驻留
 * bmm1/bmm2 以抽象调用表示（生产用 Matmul 高阶 API Iterate）
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;
constexpr float NEG_INF = -3.4e38f;

class MyFlashAttention {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                GM_ADDR softmaxMaxOut, GM_ADDR softmaxSumOut,
                                uint32_t s1, uint32_t s2, uint32_t d, GM_ADDR tiling)
    {
        this->s1 = s1; this->s2 = s2; this->d = d;
        this->dAlign = (d * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        this->s2Chunk = 128;   // //?: S2 块宽由 tiling 下发，示意
        blockIdx = AscendC::GetBlockIdx();
        qGm.SetGlobalBuffer((__gm__ float*)q, (uint64_t)s1 * d);
        kGm.SetGlobalBuffer((__gm__ float*)k, (uint64_t)s2 * d);
        vGm.SetGlobalBuffer((__gm__ float*)v, (uint64_t)s2 * d);
        oGm.SetGlobalBuffer((__gm__ float*)out, (uint64_t)s1 * d);
        mGm.SetGlobalBuffer((__gm__ float*)softmaxMaxOut, s1);
        sumGm.SetGlobalBuffer((__gm__ float*)softmaxSumOut, s1);

        pipe.InitBuffer(qBuf, 1, dAlign * sizeof(float));          // Q 行驻留
        pipe.InitBuffer(kBuf, 1, s2Chunk * sizeof(float));         // K 块
        pipe.InitBuffer(vBuf, 1, s2Chunk * dAlign * sizeof(float));// V 块（//?: 大块驻留示意，实际分块）
        pipe.InitBuffer(sBuf, 1, s2Chunk * sizeof(float));         // S 行（每轮一行 QK^T）
        pipe.InitBuffer(pBuf, 1, s2Chunk * sizeof(float));         // P 行
        pipe.InitBuffer(accBuf, 1, dAlign * sizeof(float));        // accO 行状态
        pipe.InitBuffer(stateBuf, 1, 2 * BLOCK_ALIGN);             // m/sum 行状态
        pipe.InitBuffer(vWork, 1, dAlign * sizeof(float));
        scale = 1.0f / sqrt((float)d);
    }

    __aicore__ inline void Process()
    {
        // 骨架：外层 s1 行循环 × 内层 s2 块循环（生产多核切 B/N/S1 并 extraInfo 流水化）
        AscendC::LocalTensor<float> qRow = qBuf.Get<float>();
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::LocalTensor<float> state = stateBuf.Get<float>();
        AscendC::LocalTensor<float> sRow = sBuf.Get<float>();
        AscendC::LocalTensor<float> pRow = pBuf.Get<float>();

        for (uint32_t r = 0; r < s1; r++) {
            AscendC::DataCopy(qRow, qGm[(uint64_t)r * d], dAlign);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Duplicate(acc, 0.0f, dAlign);
            float mOld = NEG_INF, sumOld = 0.0f;

            // ---- 在线 softmax：跨 s2 块维护 (m, sum, accO) ----
            for (uint32_t c = 0; c < s2; c += s2Chunk) {
                uint32_t n2 = (s2 - c) < s2Chunk ? (s2 - c) : s2Chunk;
                // S 行 = Q·K块^T（生产：bmm1.Iterate；mini 用向量点积示意）
                for (uint32_t j = 0; j < n2; j++) {
                    // dot(qRow, kRow_j) —— 实际由 bmm1 一次出整块 S
                    AscendC::Mul(vWork, qRow, kBuf.Get<float>()[j * dAlign], dAlign);  // //?: k 块载入省略
                    AscendC::ReduceSum<float>(sRow[j], vWork, vWork, dAlign);
                }
                AscendC::PipeBarrier<AscendC::PIPE_V>();
                AscendC::Muls(sRow, sRow, scale, n2);
                // 行内 max（R28 B31 折叠树/掩码计数，此处示意直读）
                AscendC::ReduceMax<float>(sRow, sRow, sRow, n2, false);
                AscendC::PipeBarrier<AscendC::PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
                float mBlock = sRow.GetValue(0);
                float mNew = mOld > mBlock ? mOld : mBlock;

                AscendC::Adds(pRow, sRow, -mNew, n2);          // P = exp(S − m_new)
                AscendC::PipeBarrier<AscendC::PIPE_V>();
                AscendC::Exp(pRow, pRow, n2);
                AscendC::PipeBarrier<AscendC::PIPE_V>();
                AscendC::ReduceSum<float>(pRow, pRow, pRow, n2);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                float cSum = pRow.GetValue(0);

                float rescale = exp(mOld - mNew);              // 旧状态重缩放（flash 核心）
                AscendC::Muls(acc, acc, rescale, dAlign);
                AscendC::PipeBarrier<AscendC::PIPE_V>();
                // accO += P · V块（生产：bmm2.Iterate；mini 逐行示意）
                for (uint32_t j = 0; j < n2; j++) {
                    float pj = pRow.GetValue(j);               // //?: 应 tensor 化——Axpy(acc, vRow_j, pj)
                    AscendC::Axpy(acc, vBuf.Get<float>()[j * dAlign], pj, dAlign);
                }
                AscendC::PipeBarrier<AscendC::PIPE_V>();
                sumOld = sumOld * rescale + cSum;
                mOld = mNew;
            }
            // ---- 归一化写出 O 与状态 ----
            AscendC::Muls(acc, acc, 1.0f / sumOld, dAlign);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::DataCopyPad(oGm[(uint64_t)r * d], acc,
                                 AscendC::DataCopyExtParams{1, (uint32_t)(d * sizeof(float)), 0, 0, 0});
            mGm.SetValue(r, mOld);
            sumGm.SetValue(r, sumOld);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qBuf, kBuf, vBuf, sBuf, pBuf, accBuf, stateBuf, vWork;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, oGm, mGm, sumGm;
    uint32_t s1, s2, d, dAlign, s2Chunk;
    float scale;
    uint16_t blockIdx;
};
