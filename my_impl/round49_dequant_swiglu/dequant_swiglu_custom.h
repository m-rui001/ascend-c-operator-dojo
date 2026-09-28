/*
 * Round 49 - 我实现的 DequantSwigluQuant mini：int8→dequant→SiLU×Mul→quant→int8
 * 数值契约链两端截断 + 双维 scale + Init 预计算倒数
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyDequantSwigluQuant {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR actScale, GM_ADDR quantScale,
                                GM_ADDR y, GM_ADDR scaleOut,
                                uint32_t rows, uint32_t colNum, GM_ADDR tiling)
    {
        this->rows = rows;
        this->colNum = colNum;
        this->colAlign = (colNum + BLOCK_ALIGN - 1) / BLOCK_ALIGN * BLOCK_ALIGN;
        blockIdx = AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ int8_t*)x, (uint64_t)rows * colNum * 2);   // gate|up 两半
        wsGm.SetGlobalBuffer((__gm__ float*)weightScale, colNum);
        asGm.SetGlobalBuffer((__gm__ float*)actScale, rows);
        qGm.SetGlobalBuffer((__gm__ float*)quantScale, colNum);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y, (uint64_t)rows * colNum);
        soGm.SetGlobalBuffer((__gm__ float*)scaleOut, (uint64_t)rows * colNum);

        pipe.InitBuffer(xQ, 2, colAlign);                       // gate/up 两半 int8
        pipe.InitBuffer(wsq, 1, colAlign * sizeof(float));
        pipe.InitBuffer(f32Buf, 4, colAlign * sizeof(float));   // 多角色 fp32 槽（配对表：见复盘）
        // Init 预计算 1/quantScale（B8：标量除数→乘倒数，一次算好）
        AscendC::LocalTensor<float> qLocal = wsq.Get<float>();
        AscendC::DataCopy(qLocal, qGm, colAlign);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        invQuantScale = 1.0f / qLocal.GetValue(0);
        // weightScale 通道级常驻载入
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> ws = wsq.Get<float>();
        AscendC::LocalTensor<float> f0 = f32Buf.Get<float>();
        AscendC::LocalTensor<float> f1 = f32Buf.Get<float>()[colAlign];
        AscendC::LocalTensor<float> f2 = f32Buf.Get<float>()[2 * colAlign];
        AscendC::LocalTensor<float> f3 = f32Buf.Get<float>()[3 * colAlign];

        for (uint32_t r = 0; r < rows; r++) {
            float actScale = asGm.GetValue(r);   // token 级标量（B42 允许级）
            // ---- 载入 gate|up 两半 int8 ----
            AscendC::LocalTensor<int8_t> gate = xQ.AllocTensor<int8_t>();
            AscendC::DataCopy(gate, xGm[(uint64_t)r * colNum * 2], colAlign);
            xQ.EnQue(gate);
            AscendC::LocalTensor<int8_t> up = xQ.AllocTensor<int8_t>();
            AscendC::DataCopy(up, xGm[(uint64_t)r * colNum * 2 + colNum], colAlign);
            xQ.EnQue(up);
            AscendC::LocalTensor<int8_t> gi = xQ.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> ui = xQ.DeQue<int8_t>();

            // ---- dequant：cast → ×weightScale(通道) → ×actScale(token) ----
            AscendC::Cast(f0, gi, AscendC::RoundMode::CAST_NONE, colNum);
            AscendC::Cast(f1, ui, AscendC::RoundMode::CAST_NONE, colNum);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Mul(f0, f0, ws, colNum);
            AscendC::Mul(f1, f1, ws, colNum);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Muls(f0, f0, actScale, colNum);
            AscendC::Muls(f1, f1, actScale, colNum);
            AscendC::PipeBarrier<AscendC::PIPE_V>();

            // ---- SiLU(gate) × up（R48 组合，fp32 域）----
            AscendC::Muls(f2, f0, -1.0f, colNum);              // 用独立槽取负（B14：不破坏原 g）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Exp(f2, f2, colNum);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Adds(f2, f2, 1.0f, colNum);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Duplicate(f3, 1.0f, colNum);
            AscendC::Div(f2, f3, f2, colNum);                  // sigmoid（Div(1,·)，B8）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Mul(f2, f0, f2, colNum);                  // SiLU(gate)
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Mul(f2, f2, f1, colNum);                  // × up

            // ---- quant：×预计算倒数 → int8 → y + scale ----
            AscendC::Muls(f2, f2, invQuantScale, colNum);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::LocalTensor<int8_t> y8 = xQ.AllocTensor<int8_t>();
            AscendC::Cast(y8, f2, AscendC::RoundMode::CAST_RINT, colNum);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopyExtParams cp{1, (uint32_t)colNum, 0, 0, 0};
            AscendC::DataCopyPad(yGm[(uint64_t)r * colNum], y8, cp);
            // scaleOut = quantScale 原值回写（fp32）
            AscendC::DataCopyPad(soGm[(uint64_t)r * colNum], qLocal, cp);
            xQ.FreeTensor(gi);
            xQ.FreeTensor(ui);
            xQ.FreeTensor(y8);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> xQ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wsq, f32Buf;
    AscendC::GlobalTensor<int8_t> xGm, yGm;
    AscendC::GlobalTensor<float> wsGm, asGm, qGm, soGm;
    uint32_t rows, colNum, colAlign;
    float invQuantScale;
    uint16_t blockIdx;
};
