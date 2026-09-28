/*
 * Round 18 - 我预测式实现的 foreach_norm（L2 范数）：走 v2 reduce 工厂形态
 * stage1: 每核把张量份额归约成 partial=Σx² 写 workspace
 * sync:   CrossCoreSetFlag/WaitFlag(PIPE_MTE3)
 * stage2: 跨步认领输出张量，读 partials 再归约，Sqrt 一次，写输出
 * 对没把握处用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t COPY_SPACE_MULTIPLE = 2;  // //?: fp16 路径多倍 UB 的倍数语义待销案

// 我的 Predicate：Norm = sqrt(Σx²)，sqrt 只出现在 round2
struct NormPred {
    __aicore__ inline void ReduceCompute1(AscendC::LocalTensor<float>& partialOut,
                                          const AscendC::LocalTensor<float>& xF32,
                                          AscendC::LocalTensor<float>& work, int64_t count) const {
        AscendC::Mul(xF32, xF32, xF32, count);            // in-place 平方（dst=src 重叠允许）
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::WholeReduceSum<float, true>(partialOut, xF32, count, 1, 1, 1, 8);  // //?: 段内归约形态
    }
    __aicore__ inline void ReduceCompute2(AscendC::LocalTensor<float>& out,
                                          const AscendC::LocalTensor<float>& partials,
                                          AscendC::LocalTensor<float>& work, int64_t count) const {
        AscendC::WholeReduceSum<float, true>(out, partials, count, 1, 1, 1, 8);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Sqrt(out, out, 1);  // //?: Sqrt 对 1 元素 tensor 的调用形态
    }
};

template <typename T, typename PRED>
class MyForeachNorm {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const ForeachReduceTilingData& tiling)
    {
        this->t = tiling;
        inTensorPtr = x; outTensorPtr = y; workTensorPtr = workspace;
        blockIdx = AscendC::GetBlockIdx();
        workGM.SetGlobalBuffer((__gm__ float*)workspace, t.middleWorkspaceSize / sizeof(float));  // //?: workspace 大小字段

        // fp16/bf16 走 fp32 中间量队列；fp32 直接算
        pipe.InitBuffer(dataQueue, BUFFER_NUM, t.inputsTensorUbSize * COPY_SPACE_MULTIPLE);
        maxDataCount = t.inputsTensorUbSize * COPY_SPACE_MULTIPLE / sizeof(T);
        pipe.InitBuffer(outQueue, 1, BYTE_BLOCK);              // 输出只是标量
        pipe.InitBuffer(calcBuf, BYTE_BLOCK);                  // 归约结果载体
        if constexpr (!std::is_same_v<T, float>) {
            pipe.InitBuffer(f32Buf, maxDataCount * sizeof(float));  // cast 中间量
        }
    }

    __aicore__ inline void Process()
    {
        PRED pred;
        // ---- Stage1: 本核张量份额 → partial 到 workspace ----
        for (uint16_t i = t.tensorStart; i <= t.tensorEnd; i++) {
            if (t.tensorDataCountList[i] == 0) continue;
            int64_t cs = (i == t.tensorStart) ? t.tensorStartOffset : 0;
            int64_t ce = (i == t.tensorEnd) ? t.tensorEndOffset : t.tensorDataCountList[i] - 1;
            int64_t count = ce - cs + 1;
            inGM.SetGlobalBuffer(GetTensorAddr(i, inTensorPtr) + cs);

            AscendC::LocalTensor<float> acc = calcBuf.Get<float>();
            AscendC::WholeReduceSum<float, true>(acc, acc, 1, 1, 1, 1, 8);  // //?: acc 清零的可靠写法存疑，改用先算后累加
            float accVal = 0.0f;  // 标量累加仅 O(段数) 且每段一次——允许级别
            for (int64_t cur = 0; cur < count; cur += maxDataCount) {
                int64_t seg = (count - cur) < maxDataCount ? (count - cur) : maxDataCount;
                CopyIn(cur, seg);
                // 段内: cast→平方→ReduceSum → 标量累加
                AscendC::LocalTensor<T> xIn = dataQueue.DeQue<T>();
                if constexpr (std::is_same_v<T, float>) {
                    AscendC::Mul(xIn, xIn, xIn, seg);
                    AscendC::PipeBarrier<AscendC::PIPE_V>();
                    AscendC::WholeReduceSum<float, true>(acc, xIn, seg, 1, 1, 1, 8);
                } else {
                    AscendC::LocalTensor<float> xF = f32Buf.Get<float>();
                    AscendC::Cast(xF, xIn, AscendC::RoundMode::CAST_NONE, seg);
                    AscendC::Mul(xF, xF, xF, seg);
                    AscendC::PipeBarrier<AscendC::PIPE_V>();
                    AscendC::WholeReduceSum<float, true>(acc, xF, seg, 1, 1, 1, 8);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);   // CHECKLIST B10
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                accVal += acc.GetValue(0);
                dataQueue.FreeTensor(xIn);
            }
            // 写 partial 到 workspace：本核在 workspace 的中间区偏移 + 张量序
            uint16_t slot = t.coreMiddleOffset + (i - t.tensorStart);  // //?: 语义待销案
            workGM.SetValue(slot, accVal);                              // //?: 标量写 GM 走什么通路
        }

        // ---- 同步全核 ----
        AscendC::CrossCoreSetFlag<0, AscendC::PIPE_MTE3>(1);
        AscendC::CrossCoreWaitFlag(1);

        // ---- Stage2: 跨步认领输出张量，读 partials 二次归约 + Sqrt ----
        for (uint16_t i = blockIdx; i < t.totalTensorCount; i += t.needCoreNum) {
            if (t.tensorDataCountList[i] == 0) { OutputZero(i); continue; }
            outGM.SetGlobalBuffer(GetTensorAddr(i, outTensorPtr), 1);
            // 读该张量的全部 partial（fp32）到 UB
            int64_t pcnt = t.tensorMiddleCountList[i];              // //?: 语义待销案
            AscendC::LocalTensor<float> pbuf = f32Buf.Get<float>();
            // //?: partials 在 workspace 的布局——按张量的区间读
            AscendC::DataCopy(pbuf, workGM[t.tensorMiddleStartList[i]], pcnt);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::LocalTensor<float> out = calcBuf.Get<float>();
            AscendC::WholeReduceSum<float, true>(out, pbuf, pcnt, 1, 1, 1, 8);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Sqrt(out, out, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            // 输出 dtype 转换 + 写出（1 元素，32B 块）
            AscendC::DataCopyExtParams cp{1, (uint32_t)(sizeof(T)), 0, 0, 0};
            if constexpr (std::is_same_v<T, float>) {
                AscendC::DataCopyPad(outGM, out, cp);
            } else {
                AscendC::LocalTensor<T> outT = outQueue.AllocTensor<T>();
                AscendC::Cast(outT, out, AscendC::RoundMode::CAST_RINT, 1);
                AscendC::DataCopyPad(outGM, outT, cp);
                outQueue.FreeTensor(outT);
            }
        }
    }

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
    {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        uint64_t off = *dataAddr;
        return reinterpret_cast<__gm__ T*>(*(dataAddr + (off >> 3) + index));
    }
    __aicore__ inline void CopyIn(int64_t off, int64_t count)
    {
        AscendC::LocalTensor<T> x = dataQueue.AllocTensor<T>();
        if (count == maxDataCount) {
            AscendC::DataCopy(x, inGM[off], count);
        } else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(x, inGM[off], cp, pad);
        }
        dataQueue.EnQue(x);
    }
    __aicore__ inline void OutputZero(uint16_t i)
    {
        outGM.SetGlobalBuffer(GetTensorAddr(i, outTensorPtr), 1);
        AscendC::LocalTensor<T> z = outQueue.AllocTensor<T>();
        AscendC::Adds(z, z, (T)0, 1);  // //?: 置零 idiom 存疑——Duplicate 更合适?
        AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(outGM, z, cp);
        outQueue.FreeTensor(z);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> dataQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf, f32Buf;
    AscendC::GlobalTensor<T> inGM, outGM;
    AscendC::GlobalTensor<float> workGM;
    GM_ADDR inTensorPtr, outTensorPtr, workTensorPtr;
    const ForeachReduceTilingData& t;
    uint16_t blockIdx;
    int64_t maxDataCount;
};
