/*
 * Round 45 - 我实现的 CrossEntropyLoss mini（fp32，LSE 稳定 + target 索引 + mean 归约）
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyCrossEntropyLoss {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR loss, GM_ADDR logProb,
                                uint32_t batchNum, uint32_t classNum, GM_ADDR tiling)
    {
        this->batchNum = batchNum;
        this->classNum = classNum;
        this->rowAlign = (classNum * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        blockIdx = AscendC::GetBlockIdx();
        uint64_t rowBase = (uint64_t)blockIdx * batchNum * classNum;
        inGm.SetGlobalBuffer((__gm__ float*)input + rowBase, (uint64_t)batchNum * classNum);
        tgGm.SetGlobalBuffer((__gm__ int*)target + blockIdx * batchNum, batchNum);
        lossGm.SetGlobalBuffer((__gm__ float*)loss + blockIdx * batchNum, batchNum);   // none 模式逐元素
        lpGm.SetGlobalBuffer((__gm__ float*)logProb + rowBase, (uint64_t)batchNum * classNum);

        pipe.InitBuffer(rowQ, 1, rowAlign * sizeof(float));       // 行驻留（三扫描共用）
        pipe.InitBuffer(tBuf, 1, rowAlign * sizeof(float));
        pipe.InitBuffer(redBuf, 1, BLOCK_ALIGN);
        pipe.InitBuffer(lossBuf, 1, ((batchNum + 7) / 8 * 8) * sizeof(float));  // loss 攒批
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> row = rowQ.Get<float>();
        AscendC::LocalTensor<float> t = tBuf.Get<float>();
        AscendC::LocalTensor<float> red = redBuf.Get<float>();
        AscendC::LocalTensor<float> lossLocal = lossBuf.Get<float>();

        float batchSumLoss = 0.0f;
        for (uint32_t b = 0; b < batchNum; b++) {
            AscendC::DataCopy(row, inGm[(uint64_t)b * classNum], rowAlign);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // ---- 1. rowMax（MIN_FLT 初始化；单次 WholeReduceMax 即可）----
            AscendC::WholeReduceMax<float, false>(red, row, classNum < 64 ? classNum : 64, 1, 8, 1, 8);  // //?: 大行需折叠树（R28 B31）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            float rowMax = red.GetValue(0);

            // ---- 2. Σexp(x − rowMax) ----
            AscendC::Adds(t, row, -rowMax, classNum);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Exp(t, t, classNum);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::ReduceSum<float>(red, t, t, classNum);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            float batchSum = red.GetValue(0);
            float logBatchSum = log(batchSum);

            // ---- 3. logProb 输出 = x − rowMax − logBatchSum（顺带第二遍扫描前的驻留计算）----
            AscendC::Adds(t, row, -rowMax - logBatchSum, classNum);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::DataCopyPad(lpGm[(uint64_t)b * classNum], t,
                                 AscendC::DataCopyExtParams{1, (uint32_t)(classNum * sizeof(float)), 0, 0, 0});

            // ---- 4. target 标量索引：loss = logBatchSum − x[target] ----
            AscendC::LocalTensor<int> tg = red.ReinterpretCast<int>();  // 借缓冲（配对表内自查：red 已消费）
            AscendC::DataCopyParams one{1, (uint16_t)sizeof(int), 0, 0};
            AscendC::DataCopyPadExtParams<int> tpad{false, 0, 0, 0};
            AscendC::DataCopyPad(tg, tgGm[b], one, tpad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID4);
            int target = tg.GetValue(0);
            float loss = logBatchSum - row.GetValue(target);   // 驻留行内随机访问（B3 索引语义）

            lossLocal.SetValue(b, loss);
            batchSumLoss += loss;   // mean 归约的标量累加（O(batch)，允许级）
        }
        // mean 模式：lossMean = batchSumLoss / batchNum 写出（none 模式逐元素写 lossLocal）
        AscendC::DataCopyExtParams cp{1, (uint32_t)(batchNum * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(lossGm, lossLocal, cp);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowQ, tBuf, redBuf, lossBuf;
    AscendC::GlobalTensor<float> inGm, lossGm, lpGm;
    AscendC::GlobalTensor<int> tgGm;
    uint32_t batchNum, classNum, rowAlign;
    uint16_t blockIdx;
};
