/*
 * Round 23 - 我实现的 foreach_lerp_scalar：y = x1 + w*(x2-x1)，含系数幅度 base-swap 分支
 * 独立类形态；fp32 直算，fp16 双槽 cast 到 fp32 域
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BYTE_BLOCK = 32;

template <typename T>
class MyLerpScalar {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y,
                                GM_ADDR workspace, const ForeachCommonTilingData* td)
    {
        this->td = td;
        blockIdx = AscendC::GetBlockIdx();
        wGm.SetGlobalBuffer((__gm__ float*)weight, 1);
        pipe.InitBuffer(wBuf, BYTE_BLOCK);
        AscendC::LocalTensor<float> wLocal = wBuf.Get<float>();
        AscendC::DataCopy(wLocal, wGm, BYTE_BLOCK / sizeof(float));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        weightVal = wLocal.GetValue(0);

        pipe.InitBuffer(q1, BUFFER_NUM, td->inputsTensorUbSize);
        pipe.InitBuffer(q2, BUFFER_NUM, td->inputsTensorUbSize);
        pipe.InitBuffer(qOut, BUFFER_NUM, td->inputsTensorUbSize);
        maxDataCount = td->inputsTensorUbSize / sizeof(T);
        if constexpr (!std::is_same_v<T, float>) {
            pipe.InitBuffer(f32q, 1, td->inputsTensorUbSize * 2);  // x1/x2 双槽 fp32
        }
    }

    __aicore__ inline void Process()
    {
        for (uint16_t idx = td->tensorStartList[blockIdx]; idx <= td->tensorEndList[blockIdx]; idx++) {
            uint64_t cs = (idx == td->tensorStartList[blockIdx]) ? td->tensorStartOffsetList[blockIdx] : 0;
            uint64_t ce = (idx == td->tensorEndList[blockIdx]) ? td->tensorEndOffsetList[blockIdx]
                                                               : td->tensorDataCountList[idx];
            g1.SetGlobalBuffer(GetAddr(idx, p1) + cs);
            g2.SetGlobalBuffer(GetAddr(idx, p2) + cs);
            gOut.SetGlobalBuffer(GetAddr(idx, pOut) + cs);
            for (uint64_t cur = cs; cur < ce; cur += maxDataCount) {
                int64_t seg = (ce - cur) < maxDataCount ? (ce - cur) : maxDataCount;
                CopyIn(cur - cs, seg);
                Compute(seg);
                CopyOut(cur - cs, seg);
            }
        }
    }

private:
    __aicore__ inline __gm__ T* GetAddr(uint16_t index, GM_ADDR tensorPtr)
    {
        __gm__ uint64_t* d = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        return reinterpret_cast<__gm__ T*>(*(d + ((*d) >> 3) + index));
    }
    __aicore__ inline void CopyIn(uint64_t off, int64_t count)
    {
        AscendC::LocalTensor<T> a = q1.AllocTensor<T>();
        AscendC::LocalTensor<T> b = q2.AllocTensor<T>();
        if (count == maxDataCount) {
            AscendC::DataCopy(a, g1[off], count);
            AscendC::DataCopy(b, g2[off], count);
        } else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(a, g1[off], cp, pad);
            AscendC::DataCopyPad(b, g2[off], cp, pad);
        }
        q1.EnQue(a); q2.EnQue(b);
    }
    __aicore__ inline void Compute(int64_t count)
    {
        AscendC::LocalTensor<T> x1 = q1.DeQue<T>();
        AscendC::LocalTensor<T> x2 = q2.DeQue<T>();
        AscendC::LocalTensor<T> out = qOut.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            // 系数幅度 base-swap（数值稳定）：|w|<=1 → x1 基；否则翻转到 x2 基
            if (weightVal <= 1.0f && weightVal >= -1.0f) {
                AscendC::Sub(x2, x2, x1, count);            // delta = x2-x1（原地）
                AscendC::PipeBarrier<AscendC::PIPE_V>();
                AscendC::Axpy(x1, x2, weightVal, count);    // x1 += w*delta
                AscendC::DataCopy(out, x1, count);           // UB 内拷出
            } else {
                AscendC::Sub(x1, x2, x1, count);            // delta' = x1-x2（原地翻转基）
                AscendC::PipeBarrier<AscendC::PIPE_V>();
                float w2 = weightVal - 1.0f;                 // |w-1| < |w|
                AscendC::Axpy(x2, x1, w2, count);            // x2 += (w-1)*delta'
                AscendC::DataCopy(out, x2, count);
            }
        } else {
            AscendC::LocalTensor<float> f32 = f32q.Get<float>();
            AscendC::Cast(f32, x1, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(f32[maxDataCount], x2, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Sub(f32[maxDataCount], f32[maxDataCount], f32, count);
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Axpy(f32, f32[maxDataCount], weightVal, count);  // fp32 域 Axpy（weight 精度不降）
            AscendC::PipeBarrier<AscendC::PIPE_V>();
            AscendC::Cast(out, f32, AscendC::RoundMode::CAST_RINT, count);
        }
        qOut.EnQue(out);
        q1.FreeTensor(x1);
        q2.FreeTensor(x2);
    }
    __aicore__ inline void CopyOut(uint64_t off, int64_t count)
    {
        AscendC::LocalTensor<T> out = qOut.DeQue<T>();
        if (count == maxDataCount) {
            AscendC::DataCopy(gOut[off], out, count);
        } else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(gOut[off], out, cp);
        }
        qOut.FreeTensor(out);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> q1, q2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> qOut;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wBuf, f32q;
    AscendC::GlobalTensor<T> g1, g2, gOut;
    AscendC::GlobalTensor<float> wGm;
    GM_ADDR p1, p2, pOut;
    const ForeachCommonTilingData* td;
    float weightVal;
    uint16_t blockIdx;
    int64_t maxDataCount;
};
