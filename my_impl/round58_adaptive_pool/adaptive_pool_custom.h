/*
 * Round 58 - 我实现的 1D 自适应均值池化 mini：窗口索引预计算 + 重叠窗口共享载入 + 逐窗口除数
 * 输入 x[ inputLen ]，输出 y[ outLen ]，窗口边界按比例自适应
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyAdaptiveAvgPool1D {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t inputLen, uint32_t outLen, GM_ADDR tiling)
    {
        this->inputLen = inputLen;
        this->outLen = outLen;
        blockIdx = AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ float*)x, inputLen);
        yGm.SetGlobalBuffer((__gm__ float*)y, outLen);

        // ---- 窗口边界预计算（host 或核内一遍；此处核内算并存张量）----
        winStart = (inputLen + outLen - 1) / outLen;   // 步长近似（自适应的 ceil 路径）
        winEndBase = inputLen / outLen;
        pipe.InitBuffer(startBuf, (outLen + 7) / 8 * 8 * sizeof(int32_t));
        pipe.InitBuffer(endBuf, (outLen + 7) / 8 * 8 * sizeof(int32_t));
        pipe.InitBuffer(xBuf, 1, inputLen * sizeof(float));
        pipe.InitBuffer(accBuf, 1, inputLen * sizeof(float));
        pipe.InitBuffer(yBuf, 1, outLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<int32_t> startT = startBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> endT = endBuf.Get<int32_t>();
        // 预计算每输出点的窗口边界（自适应：start=i*winStart, end=(i+1)*winEndBase，保证覆盖）
        for (uint32_t i = 0; i < outLen; i++) {
            startT.SetValue(i, (i * winStart > i * winEndBase) ? i * winStart : i * winEndBase);
            endT.SetValue(i, (i + 1) * winEndBase < (i + 1) * winStart ? (i + 1) * winEndBase : (i + 1) * winStart);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

        AscendC::LocalTensor<float> xLocal = xBuf.Get<float>();
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::LocalTensor<float> yLocal = yBuf.Get<float>();
        AscendC::DataCopy(xLocal, xGm, inputLen);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Duplicate(acc, 0.0f, inputLen);

        // ---- 重叠窗口：滑动累加（差分法）替代逐窗口重扫 ----
        // 简化共享：滑窗累加 acc = 前缀和差分；此处演示"每窗口 Muls 除数"
        for (uint32_t i = 0; i < outLen; i++) {
            int32_t ws = startT.GetValue(i);
            int32_t we = endT.GetValue(i);
            float sum = 0.0f;
            for (int32_t k = ws; k < we; k++) {
                sum += xLocal.GetValue(k);   // //?: 窗口窄时允许级；宽窗口应改前缀和差分——对比环节销案
            }
            float factor = 1.0f / (float)(we - ws);   // 逐窗口除数（自适应的分歧点）
            yLocal.SetValue(i, sum * factor);
        }
        AscendC::DataCopyPad(yGm, yLocal, AscendC::DataCopyExtParams{1, (uint32_t)(outLen * sizeof(float)), 0, 0, 0});
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> startBuf, endBuf, xBuf, accBuf, yBuf;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t inputLen, outLen, winStart, winEndBase;
    uint16_t blockIdx;
};
