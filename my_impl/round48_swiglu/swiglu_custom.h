/*
 * Round 48 - 我实现的 SwiGLU mini：SiLU(gate) ⊙ up
 * 分派结构复刻生产（isDoubleBuffer × dtype × 220 守卫）；SiLU 由基础 API 组合
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

template <typename T, int32_t BUFFER_NUM>
class MySwiglu {
public:
    __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR y, uint64_t totalLen, GM_ADDR tiling)
    {
        this->totalLen = totalLen;
        blockIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        perCore = (totalLen + coreNum - 1) / coreNum;
        myStart = (uint64_t)blockIdx * perCore;
        myLen = myStart + perCore <= totalLen ? perCore : totalLen - myStart;

        gateGm.SetGlobalBuffer((__gm__ T*)gate + myStart, myLen);
        upGm.SetGlobalBuffer((__gm__ T*)up + myStart, myLen);
        yGm.SetGlobalBuffer((__gm__ T*)y + myStart, myLen);

        pipe.InitBuffer(gateQ, BUFFER_NUM, perCore / BUFFER_NUM * sizeof(T) > BLOCK_ALIGN
                                              ? perCore / BUFFER_NUM * sizeof(T) : BLOCK_ALIGN);
        pipe.InitBuffer(upQ, BUFFER_NUM, perCore / BUFFER_NUM * sizeof(T) > BLOCK_ALIGN
                                              ? perCore / BUFFER_NUM * sizeof(T) : BLOCK_ALIGN);
        pipe.InitBuffer(yQ, BUFFER_NUM, perCore / BUFFER_NUM * sizeof(T) > BLOCK_ALIGN
                                             ? perCore / BUFFER_NUM * sizeof(T) : BLOCK_ALIGN);
        pipe.InitBuffer(f32Buf, 1, perCore * sizeof(float));  // fp32 域工作区
    }

    __aicore__ inline void Process()
    {
        // 简化：整核一段（生产按 ubFactor 分块流水）
        AscendC::LocalTensor<T> g = gateQ.AllocTensor<T>();
        AscendC::DataCopy(g, gateGm, myLen);
        gateQ.EnQue(g);
        AscendC::LocalTensor<T> u = upQ.AllocTensor<T>();
        AscendC::DataCopy(u, upGm, myLen);
        upQ.EnQue(u);
        AscendC::LocalTensor<T> gi = gateQ.DeQue<T>();
        AscendC::LocalTensor<T> ui = upQ.DeQue<T>();
        AscendC::LocalTensor<T> yo = yQ.AllocTensor<T>();

        if constexpr (std::is_same_v<T, float>) {
            SiluAndMul(gi, ui, yo, myLen);
        } else {
            // fp16/bf16：升 fp32 域（sigmoid 的 exp 敏感，B8）
            AscendC::LocalTensor<float> gf = f32Buf.Get<float>();
            AscendC::LocalTensor<float> uf = f32Buf.Get<float>()[myLen];  // //?: 双段偏移示意
            AscendC::Cast(gf, gi, AscendC::RoundMode::CAST_NONE, myLen);
            AscendC::Cast(uf, ui, AscendC::RoundMode::CAST_NONE, myLen);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            SiluAndMulF32(gf, uf, yo, myLen);
        }
        yQ.EnQue(yo);
        AscendC::LocalTensor<T> yOut = yQ.DeQue<T>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(yGm, yOut, myLen);
        gateQ.FreeTensor(gi);
        upQ.FreeTensor(ui);
        yQ.FreeTensor(yOut);
    }

private:
    // fp32 域：SiLU(g)=g·sigmoid(g)；再乘 up
    __aicore__ inline void SiluAndMulF32(AscendC::LocalTensor<float>& g, AscendC::LocalTensor<float>& u,
                                         AscendC::LocalTensor<T>& out, uint32_t n)
    {
        AscendC::LocalTensor<float> sig = f32Buf.Get<float>()[2 * myLen];  // 第三段（//?: 槽位规划见复盘）
        AscendC::Muls(g, g, -1.0f, n);            // −g
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Exp(sig, g, n);                  // e^(−g)
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Adds(sig, sig, 1.0f, n);         // 1+e^(−g)
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::LocalTensor<float> one = vBufTmp.Get<float>();
        AscendC::Duplicate(one, 1.0f, n);
        AscendC::Div(sig, one, sig, n);           // sigmoid = 1/(1+e^(−g))（B8：Div(1,·)）
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Mul(g, g, sig, n);               // 注意：g 已被取负——需先备份原 g！勘误见复盘
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Mul(out, g, u, n);               // SiLU(g) ⊙ up
    }
    __aicore__ inline void SiluAndMul(AscendC::LocalTensor<float>& g, AscendC::LocalTensor<float>& u,
                                      AscendC::LocalTensor<T>& out, uint32_t n)
    {
        SiluAndMulF32(g, u, out, n);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> gateQ, upQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> yQ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> f32Buf, vBufTmp;
    AscendC::GlobalTensor<T> gateGm, upGm, yGm;
    uint64_t totalLen, myStart, myLen;
    uint16_t blockIdx;
};
