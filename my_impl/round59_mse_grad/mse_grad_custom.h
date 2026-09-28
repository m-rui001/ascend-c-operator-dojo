/*
 * Round 59 - 我实现的 MSELossGrad mini：三输入单队列打包 + cof 融合 + 尾核降级
 * dx = (predict − label) × cof × dout
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyMseGrad {
public:
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR dx,
                                uint64_t totalLen, float cof, uint32_t tilingBufferNum, GM_ADDR tiling)
    {
        this->cof = cof;
        this->totalLen = totalLen;
        this->bufferNum = tilingBufferNum;   // usedDb：host 决定 1 或 2（B7 机制化）
        uint32_t coreNum = AscendC::GetBlockNum();
        perCore = (totalLen + coreNum - 1) / coreNum;
        myStart = (uint64_t)blockIdx * perCore;
        myLen = myStart + perCore <= totalLen ? perCore : totalLen - myLen0(myStart, perCore, totalLen);
        isTailPad = (myLen % FLOATS_PER_TILE) != 0;   // //?: 尾核判定按 tiling padLength，示意

        xGm.SetGlobalBuffer((__gm__ float*)predict + myStart, myLen);
        yGm.SetGlobalBuffer((__gm__ float*)label + myStart, myLen);
        zGm.SetGlobalBuffer((__gm__ float*)dout + myStart, myLen);
        oGm.SetGlobalBuffer((__gm__ float*)dx + myStart, myLen);

        chunk = 2048;   // //?: tileLength 由 tiling 下发，示意
        // 三输入单队列：槽 = chunk × 3
        pipe.InitBuffer(inQ, bufferNum, chunk * 3 * sizeof(float));
        pipe.InitBuffer(outQ, bufferNum, chunk * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t tiles = (myLen + chunk - 1) / chunk;
        // 尾核降级：pad 情况切单 tile（生产 tail 核 tileNum=1/bufferNum=1）
        if (isTailPad) tiles = (myLen + chunk - 1) / chunk;
        for (uint32_t t = 0; t < tiles; t++) {
            uint64_t off = (uint64_t)t * chunk;
            uint32_t n = (myLen - off) < chunk ? (uint32_t)(myLen - off) : chunk;
            CopyIn(off, n);
            Compute(n);
            CopyOut(off, n);
        }
    }

private:
    __aicore__ inline uint64_t myLen0(uint64_t s, uint32_t p, uint64_t t) { return s + p <= t ? p : t - s; }
    __aicore__ inline void CopyIn(uint64_t off, uint32_t n)
    {
        AscendC::LocalTensor<float> packed = inQ.AllocTensor<float>();
        uint32_t aligned = (n + 7) / 8 * 8;
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyExtParams cp{1, (uint32_t)(n * sizeof(float)), 0, 0, 0};
        // 三输入写同槽三段（[0]/[chunk]/[2*chunk]）
        AscendC::DataCopyPad(packed, xGm[off], cp, pad);
        AscendC::DataCopyPad(packed[chunk], yGm[off], cp, pad);
        AscendC::DataCopyPad(packed[2 * chunk], zGm[off], cp, pad);
        inQ.EnQue(packed);
    }
    __aicore__ inline void Compute(uint32_t n)
    {
        AscendC::LocalTensor<float> packed = inQ.DeQue<float>();
        AscendC::LocalTensor<float> xSeg = packed;
        AscendC::LocalTensor<float> ySeg = packed[chunk];
        AscendC::LocalTensor<float> zSeg = packed[2 * chunk];
        AscendC::LocalTensor<float> out = outQ.AllocTensor<float>();
        AscendC::Sub(out, xSeg, ySeg, n);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Muls(out, out, cof, n);          // cof=2/N host 算（C5）
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Mul(out, out, zSeg, n);          // × dout
        outQ.EnQue(out);
        inQ.FreeTensor(packed);
    }
    __aicore__ inline void CopyOut(uint64_t off, uint32_t n)
    {
        AscendC::LocalTensor<float> out = outQ.DeQue<float>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(oGm[off], out, AscendC::DataCopyExtParams{1, (uint32_t)(n * sizeof(float)), 0, 0, 0});
        outQ.FreeTensor(out);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQ;
    AscendC::GlobalTensor<float> xGm, yGm, zGm, oGm;
    uint64_t totalLen, myStart, perCore;
    uint32_t myLen, bufferNum, chunk;
    bool isTailPad;
    float cof;
    uint16_t blockIdx;
    static constexpr uint32_t FLOATS_PER_TILE = 8;
};
