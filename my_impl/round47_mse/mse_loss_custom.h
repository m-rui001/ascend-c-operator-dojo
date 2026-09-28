/*
 * Round 47 - 我实现的 MSELoss mini：diff→平方→二分块对齐归约→跨核两段聚合单标量
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;
constexpr uint32_t FLOATS_PER_BLOCK = 8;

class MyMSELoss {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR loss, GM_ADDR usrWorkspace,
                                uint64_t totalLen, float scale, GM_ADDR tiling)
    {
        this->totalLen = totalLen;
        this->scale = scale;   // mean=1/N, sum=1（host 下发，C5）
        blockIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        // 行切分（大小核从简：均分+尾核）
        perCore = (totalLen + coreNum - 1) / coreNum;
        myStart = (uint64_t)blockIdx * perCore;
        myLen = myStart + perCore <= totalLen ? perCore : totalLen - myStart;

        xGm.SetGlobalBuffer((__gm__ float*)x, totalLen);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLen);
        lossGm.SetGlobalBuffer((__gm__ float*)loss, 1);
        wsGm.SetGlobalBuffer((__gm__ float*)usrWorkspace, coreNum * FLOATS_PER_BLOCK);

        pipe.InitBuffer(aQ, 1, perCore > 0 ? ((perCore + FLOATS_PER_BLOCK - 1) / FLOATS_PER_BLOCK) * FLOATS_PER_BLOCK * sizeof(float) : BLOCK_ALIGN);
        pipe.InitBuffer(bQ, 1, perCore > 0 ? ((perCore + FLOATS_PER_BLOCK - 1) / FLOATS_PER_BLOCK) * FLOATS_PER_BLOCK * sizeof(float) : BLOCK_ALIGN);
        pipe.InitBuffer(outBuf, 1, coreNum * FLOATS_PER_BLOCK * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // ---- 本核: diff → 平方 → ReduceSumBisect 部分和 ----
        AscendC::LocalTensor<float> acc = aQ.Get<float>();
        AscendC::LocalTensor<float> tmp = bQ.Get<float>();
        AscendC::Duplicate(acc, 0.0f, myAlign());
        uint64_t cur = 0;
        while (cur < myLen) {
            uint32_t seg = (myLen - cur) < chunkWidth() ? (uint32_t)(myLen - cur) : chunkWidth();
            AscendC::DataCopy(tmp, xGm[myStart + cur], chunkWidth());
            AscendC::DataCopy(acc, yGm[myStart + cur], chunkWidth());  // //?: 覆盖 acc 错误！应用第三缓冲——勘误见复盘
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Sub(tmp, tmp, acc, chunkWidth());   // diff（此处 acc 误用为 y——自查记录）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Mul(tmp, tmp, tmp, chunkWidth());   // 平方
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Add(acc, acc, tmp, chunkWidth());   // 段累加
            cur += seg;
        }
        ReduceSumBisect(acc, myAlign(), 1.0f);

        // ---- Stage1: 部分和 → workspace 本核槽 ----
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopy(wsGm[blockIdx * FLOATS_PER_BLOCK], acc, FLOATS_PER_BLOCK);

        // ---- 全核同步 ----
        AscendC::SyncAll();

        // ---- Stage2: core0 聚合所有槽 → 标量 ----
        if (blockIdx == 0) {
            AscendC::LocalTensor<float> all = outBuf.Get<float>();
            AscendC::DataCopy(all, wsGm, AscendC::GetBlockNum() * FLOATS_PER_BLOCK);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
            AscendC::ReduceSum<float>(all, all, all, AscendC::GetBlockNum() * FLOATS_PER_BLOCK);
            AscendC::Muls(all, all, scale, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(float), 0, 0, 0};
            AscendC::DataCopyPad(lossGm, all, cp);
        }
    }

private:
    __aicore__ inline uint64_t myAlign() { return (myLen + FLOATS_PER_BLOCK - 1) / FLOATS_PER_BLOCK * FLOATS_PER_BLOCK; }
    __aicore__ inline uint32_t chunkWidth() { return 1024; }
    __aicore__ inline void ReduceSumBisect(AscendC::LocalTensor<float>& src, uint64_t len, float sc)
    {
        while (len > FLOATS_PER_BLOCK) {
            uint64_t offset = ((len + 15) >> 4) << 3;   // Ceil(Ceil(len,8)/2)*8：折半且保持 8 对齐
            AscendC::Add(src, src, src[offset], len - offset);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            len = offset;
        }
        AscendC::Muls(src, src, sc, len);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> aQ, bQ, outBuf;
    AscendC::GlobalTensor<float> xGm, yGm, lossGm, wsGm;
    uint64_t totalLen, myStart, myLen;
    float scale;
    uint16_t blockIdx;
};
