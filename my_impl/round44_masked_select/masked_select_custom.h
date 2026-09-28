/*
 * Round 44 - 我实现的 MaskedSelect mini：mask→位图→bit-packed GatherMask 压缩→动态长度写出
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyMaskedSelect {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR shapeOut,
                                uint32_t totalLen, GM_ADDR tiling)
    {
        this->totalLen = totalLen;
        this->tileLen = (totalLen + 63) / 64 * 64;   // //?: tile 由 tiling 下发，示意
        blockIdx = AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLen);
        maskGm.SetGlobalBuffer((__gm__ uint8_t*)mask, totalLen);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLen);          // 上界预分配
        shapeGm.SetGlobalBuffer((__gm__ int*)shapeOut, 1);

        pipe.InitBuffer(xQ, 1, tileLen * sizeof(float));
        pipe.InitBuffer(maskQ, 1, tileLen * sizeof(uint8_t));
        pipe.InitBuffer(bitBuf, 1, tileLen / 8 * sizeof(uint8_t));   // 位图 1/8
        pipe.InitBuffer(yQ, 1, tileLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> xLocal = xQ.Get<float>();
        AscendC::LocalTensor<uint8_t> maskLocal = maskQ.Get<uint8_t>();
        AscendC::LocalTensor<uint32_t> bitMask = bitBuf.Get<uint32_t>();
        AscendC::LocalTensor<float> yLocal = yQ.Get<float>();

        AscendC::DataCopy(xLocal, xGm, tileLen);
        AscendC::DataCopyExtParams mcp{1, (uint32_t)totalLen, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> mpad{false, 0, 0, 0};
        AscendC::DataCopyPad(maskLocal, maskGm, mcp, mpad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // ---- mask → 位图：uint8(0/1) cast 到 uint32 视角的比较位 ----
        // 生产用 Cast+CompareScalar(EQ 1.0) 生成；fp32 路径直接 CompareScalar
        AscendC::LocalTensor<float> maskF32 = yLocal;   // 借 y 缓冲做比较位（配对表内自查：y 此时尚未写入）
        AscendC::Cast(maskF32, maskLocal, AscendC::RoundMode::CAST_NONE, totalLen);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::CompareScalar(bitMask.ReinterpretCast<float>(), maskF32, 1.0f,
                               AscendC::CMPMODE::EQ, totalLen);   // //?: Compare 输出位图的确切视角转换待销案
        AscendC::PipeBarrier<AscendC::PIPE_V>();

        // ---- bit-packed 压缩：rsvdCnt 即动态输出长度 ----
        uint64_t rsvdCnt = 0;
        AscendC::GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = 1;
        params.src0RepeatStride = 8;
        params.src1RepeatStride = 1;
        AscendC::GatherMask(yLocal, xLocal, bitMask, true, totalLen, params, rsvdCnt);
        AscendC::PipeBarrier<AscendC::PIPE_V>();

        // ---- 写出 y（前 rsvdCnt 个）与动态 shape ----
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(rsvdCnt * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm, yLocal, cp);
        AscendC::LocalTensor<int> shapeLocal = xQ.Get<int>();  // 借缓冲写 shape
        shapeLocal.SetValue(0, (int)rsvdCnt);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::DataCopyExtParams sp{1, (uint32_t)sizeof(int), 0, 0, 0};
        AscendC::DataCopyPad(shapeGm, shapeLocal, sp);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQ_unused, maskQ_unused;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xQ, maskQ, bitBuf, yQ;
    AscendC::GlobalTensor<float> xGm, yGm;
    AscendC::GlobalTensor<uint8_t> maskGm;
    AscendC::GlobalTensor<int> shapeGm;
    uint32_t totalLen, tileLen;
    uint16_t blockIdx;
};
