/*
 * Round 56 - 我实现的 Dropout 位掩码解包 mini：GM 1bit/元素 → Select 展开 → GM 1byte/元素
 * 对偶关系：R44 位图压缩(GatherMask) vs 本轮位图解包(Select)
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyDropMaskUnpack {
public:
    __aicore__ inline void Init(GM_ADDR bitMask, GM_ADDR byteMask,
                                uint64_t totalElems, uint32_t tilingChunk, GM_ADDR tiling)
    {
        this->totalElems = totalElems;
        this->chunk = tilingChunk;                 // 单次 UB 计算量（tiling 下发）
        blockIdx = AscendC::GetBlockIdx();
        coreNum = AscendC::GetBlockNum();
        perCore = (totalElems + coreNum - 1) / coreNum;
        myStart = (uint64_t)blockIdx * perCore;
        myLen = myStart + perCore <= totalElems ? perCore : totalElems - myStart;

        bitGm.SetGlobalBuffer((__gm__ uint8_t*)bitMask, (totalElems + 7) / 8);
        byteGm.SetGlobalBuffer((__gm__ uint8_t*)byteMask, totalElems);

        pipe.InitBuffer(bitQ, 1, (chunk + 7) / 8 * sizeof(uint8_t));   // 位图 x/8 字节
        pipe.InitBuffer(selSrcBuf, 1, chunk * sizeof(half));            // select 源（全 1）
        pipe.InitBuffer(selResBuf, 1, chunk * sizeof(half));            // select 结果
        pipe.InitBuffer(outBuf, 1, chunk * sizeof(uint8_t));            // bool 输出

        // select 源常量：全 1（保留位）
        AscendC::LocalTensor<half> src = selSrcBuf.Get<half>();
        AscendC::Duplicate(src, 1.0f, chunk);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<uint8_t> bits = bitQ.Get<uint8_t>();
        AscendC::LocalTensor<half> selSrc = selSrcBuf.Get<half>();
        AscendC::LocalTensor<half> selRes = selResBuf.Get<half>();
        AscendC::LocalTensor<uint8_t> outB = outBuf.Get<uint8_t>();

        uint64_t done = 0;
        while (done < myLen) {
            uint32_t n = (myLen - done) < chunk ? (uint32_t)(myLen - done) : chunk;
            uint32_t bytes = (n + 7) / 8;
            // 位图进 UB
            AscendC::DataCopyExtParams bcp{1, (uint32_t)bytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<uint8_t> bpad{false, 0, 0, 0};
            AscendC::DataCopyPad(bits, bitGm[(myStart + done) / 8], bcp, bpad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // ---- Select 位解包：src1(位图) stride=0 反复读，dst stride=8 展开 ----
            AscendC::BinaryRepeatParams bp;
            bp.src0BlkStride = 1;
            bp.src0RepStride = 0;      // src0（全 1 常量）反复读同一块
            bp.src1BlkStride = 1;
            bp.src1RepStride = 0;      // src1（位图）反复读——Select 的掩码语义
            bp.dstBlkStride = 1;
            bp.dstRepStride = 8;       // 输出逐 256B 推进
            AscendC::Select(selRes, selSrc, selSrc, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n, bp);
            // //?: 位选择的确切展开链（生产经 Cast+ShiftLeft+Add 组位）——示意级，跨轮销案
            AscendC::PipeBarrier<AscendC::PIPE_V>();

            // half 结果 → uint8 bool
            AscendC::Cast(outB, selRes, AscendC::RoundMode::CAST_RINT, n);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::DataCopyExtParams ocp{1, (uint32_t)n, 0, 0, 0};
            AscendC::DataCopyPad(byteGm[myStart + done], outB, ocp);
            done += n;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bitQ, selSrcBuf, selResBuf, outBuf;
    AscendC::GlobalTensor<uint8_t> bitGm, byteGm;
    uint64_t totalElems, myStart, myLen;
    uint32_t chunk, coreNum;
    uint16_t blockIdx;
};
