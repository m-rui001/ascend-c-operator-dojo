/*
 * Round 28 - 我实现的动态量化块：y(fp32 驻留) → Abs → strided Max 折叠 → scale=127/max → 广播除 → int8
 * 关键：快路径行最大值（生产 ReduceMaxInplace 同型）+ 量化输出对（int8 + fp32 scale）
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;
constexpr float QUANT_DIVIDEND = 127.0f;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;  // 256B/4B

class MyDynamicQuant {
public:
    __aicore__ inline void Init(GM_ADDR yIn, GM_ADDR scaleOut, GM_ADDR yInt8Out, GM_ADDR tiling)
    {
        this->rowSize = tdRowSize;
        this->rowsPerCore = tdRowsPerCore;
        this->rowAlign = (rowSize + ELEM_PER_REP_FP32 - 1) / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32;
        blockIdx = AscendC::GetBlockIdx();
        yGm.SetGlobalBuffer((__gm__ float*)yIn + (uint64_t)blockIdx * rowsPerCore * rowSize,
                            (uint64_t)rowsPerCore * rowSize);
        i8Gm.SetGlobalBuffer((__gm__ int8_t*)yInt8Out + (uint64_t)blockIdx * rowsPerCore * rowSize,
                             (uint64_t)rowsPerCore * rowSize);
        scaleGm.SetGlobalBuffer((__gm__ float*)scaleOut + blockIdx * rowsPerCore, rowsPerCore);

        pipe.InitBuffer(yBuf, 1, rowsPerCore * rowAlign * sizeof(float));   // 块驻留（两遍用）
        pipe.InitBuffer(vBuf, 1, rowAlign * sizeof(float));                 // abs 载体
        pipe.InitBuffer(oneBuf, 1, BLOCK_ALIGN);
        pipe.InitBuffer(scaleBuf, 1, rowsPerCore * sizeof(float));          // scale 攒批
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> yLocal = yBuf.Get<float>();
        AscendC::LocalTensor<float> vLocal = vBuf.Get<float>();
        AscendC::LocalTensor<float> sLocal = scaleBuf.Get<float>();
        // 载入整块（2D pad）
        AscendC::DataCopyExtParams cp{rowsPerCore, (uint32_t)(rowSize * sizeof(float)), 0,
                                      (uint16_t)((rowAlign - rowSize) * sizeof(float) / BLOCK_ALIGN), 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(yLocal, yGm, cp, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t r = 0; r < rowsPerCore; r++) {
            uint32_t ro = r * rowAlign;
            // ---- Abs 到独立载体（保 y 原值供除法）----
            AscendC::Abs(vLocal, yLocal[ro], rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            // ---- strided in-place Max 折叠（生产快路径同型）----
            float rowMax = MaxTreeInplace(vLocal, rowSize);
            // ---- scale = 127/max（标量除法，每行一次）----
            float scale = QUANT_DIVIDEND / rowMax;
            sLocal.SetValue(r, scale);
            // ---- 广播除法：y *= 1/scale（乘倒数，R20/B8 标量除数）----
            AscendC::Muls(yLocal[ro], yLocal[ro], 1.0f / scale, rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            // ---- int8 写出 ----
            AscendC::LocalTensor<half> halfTmp = vBuf.Get<half>();  // //?: 载体 dtype 切换依赖 vBuf 生命周期，示意
            AscendC::Cast(halfTmp, yLocal[ro], AscendC::RoundMode::CAST_RINT, rowSize);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::LocalTensor<int8_t> i8Tmp = halfTmp.ReinterpretCast<int8_t>();
            AscendC::Cast(i8Tmp, halfTmp, AscendC::RoundMode::CAST_RINT, rowSize);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::DataCopyExtParams c8{1, (uint32_t)rowSize, 0, 0, 0};
            AscendC::DataCopyPad(i8Gm[r * rowSize], i8Tmp, c8);
        }
        // scale 攒批写出（fp32）
        AscendC::DataCopyExtParams cs{1, (uint32_t)(rowsPerCore * sizeof(float)), 0, 0, 0};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::DataCopyPad(scaleGm, sLocal, cs);
    }

private:
    // 快路径行 max：repeat 内 64 元素分组，dstRepStride=0 反复写同块，srcRepStride=8 逐组推进（生产同型）
    __aicore__ inline float MaxTreeInplace(AscendC::LocalTensor<float>& v, uint32_t count)
    {
        uint32_t reps = count >> 6;
        uint32_t offs = reps << 6;
        uint32_t rems = count & 0x3f;
        if (reps > 1) {
            AscendC::Max(v, v[ELEM_PER_REP_FP32], v, ELEM_PER_REP_FP32, reps - 1, {1, 1, 1, 0, 8, 0});
            AscendC::PipeBarrier<AscendC::PIPE_V>();
        }
        if (rems > 0) {
            AscendC::Max(v, v[offs], v, rems, 1, {1, 1, 1, 0, 8, 0});
            AscendC::PipeBarrier<AscendC::PIPE_V>();
        }
        uint32_t mask = (reps > 0) ? ELEM_PER_REP_FP32 : count;
        AscendC::WholeReduceMax<float, false>(v, v, mask, 1, 8, 1, 8);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        return v.GetValue(0);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> yBuf, vBuf, oneBuf, scaleBuf;
    AscendC::GlobalTensor<float> yGm, scaleGm;
    AscendC::GlobalTensor<int8_t> i8Gm;
    uint32_t rowSize, rowsPerCore, rowAlign, tdRowSize, tdRowsPerCore;
    uint16_t blockIdx;
};
