/*
 * Round 53 - 我实现的 FA Grad Post mini：workspace fp32 部分和 → rescale → cast → 最终输出
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;
constexpr uint32_t FLOATS_PER_BLOCK = 8;

class MyFAPost {
public:
    __aicore__ inline void Init(GM_ADDR workspace, GM_ADDR dqOut,
                                uint64_t dqWsLen, uint64_t dkWsLen, uint64_t dvWsLen,
                                float rescale, GM_ADDR tiling)
    {
        this->rescale = rescale;   // 含 1/softmaxSum 与 scaleValue 的合并系数（host 预算）
        blockIdx = AscendC::GetBlockIdx();
        coreNum = AscendC::GetBlockNum();
        // workspace 三段偏移（dq|dk|dv）
        dqWs.SetGlobalBuffer((__gm__ float*)workspace, dqWsLen);
        dkWs.SetGlobalBuffer((__gm__ float*)workspace + dqWsLen, dkWsLen);
        dvWs.SetGlobalBuffer((__gm__ float*)workspace + dqWsLen + dkWsLen, dvWsLen);
        dqOutGm.SetGlobalBuffer((__gm__ half*)dqOut, dqWsLen);
        dkOutGm.SetGlobalBuffer((__gm__ half*)dkOut, 0);
        dvOutGm.SetGlobalBuffer((__gm__ half*)dv, 0);
        (void)dkOutGm; (void)dvOutGm;

        // 核内分块：每核负责一段输出
        perCore = (dqWsLen + coreNum - 1) / coreNum;
        myStart = (uint64_t)blockIdx * perCore;
        myLen = myStart + perCore <= dqWsLen ? perCore : dqWsLen - myStart;

        pipe.InitBuffer(inQ, 1, CHUNK * sizeof(float));
        pipe.InitBuffer(outQ, 1, CHUNK * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        // dq 段：读 fp32 部分和 → Muls(rescale) → Cast(ROUND) → half 写出
        AscendC::LocalTensor<float> fin = inQ.Get<float>();
        AscendC::LocalTensor<half> fout = outQ.Get<half>();
        for (uint64_t i = myStart; i < myStart + myLen; i += CHUNK) {
            uint32_t n = (myStart + myLen - i) < CHUNK ? (uint32_t)(myStart + myLen - i) : CHUNK;
            AscendC::DataCopy(fin, dqWs[i], (n + FLOATS_PER_BLOCK - 1) / FLOATS_PER_BLOCK * FLOATS_PER_BLOCK);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Muls(fin, fin, rescale, n);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Cast(fout, fin, AscendC::RoundMode::CAST_ROUND, n);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            // fp16 输出对齐 16 元素（32B）；fp32 对齐 8（R53 新知：cast 写出粒度随 dtype）
            AscendC::DataCopyExtParams cp{1, (uint32_t)(n * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPad(dqOutGm[i], fout, cp);
        }
        // dk/dv 段同构（省略）
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inQ, outQ;
    AscendC::GlobalTensor<float> dqWs, dkWs, dvWs;
    AscendC::GlobalTensor<half> dqOutGm, dkOutGm, dvOutGm;
    uint64_t dqWsLen, dkWsLen, dvWsLen, myStart, myLen;
    static constexpr uint32_t CHUNK = 1024;
    float rescale;
    uint16_t blockIdx, coreNum;
};
