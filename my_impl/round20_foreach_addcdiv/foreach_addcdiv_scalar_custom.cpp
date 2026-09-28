/*
 * Round 20 - 我实现的 foreach_addcdiv_scalar：y[i] = x1[i] + scalar*(x2[i]/x3[i])
 * 走 QuaternaryImplictOutput 工厂形态：三列表走表 + 结果累积进 x1 本地缓冲 + Div 直除 + Axpy
 * 对没把握处用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BYTE_BLOCK = 32;

// ---- 我的 Adapter（fp32 路径）----
// 签名对齐 OneScalarQuaternaryImplictOutputOp：结果累积进 tensor1（隐式输出）
template <typename T>
__aicore__ void MyAddcDivAdapter(
    const AscendC::LocalTensor<T>& tensor1Local, const AscendC::LocalTensor<T>& tensor2Local,
    const AscendC::LocalTensor<T>& tensor3Local, const AscendC::LocalTensor<float>& float32Tensor,
    const float scalarVal, const uint32_t maxCastDataCount, const int64_t dataCount)
{
    if constexpr (std::is_same_v<T, float>) {
        // 矢量÷矢量：Div 直除一条指令（B8 精化：Reciprocal 倒数只适用于标量/重复除数）
        AscendC::Div(tensor2Local, tensor2Local, tensor3Local, dataCount);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Axpy<T, T>(tensor1Local, tensor2Local, scalarVal, dataCount);  // x1 += scalar*(x2/x3)
    } else {
        // fp16/bf16：cast 到 fp32 中间量做除法再 cast 回，最后 Axpy 在原精度
        AscendC::Cast(float32Tensor, tensor2Local, AscendC::RoundMode::CAST_NONE, dataCount);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Cast(float32Tensor[maxCastDataCount], tensor3Local, AscendC::RoundMode::CAST_NONE, dataCount);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Div(float32Tensor, float32Tensor, float32Tensor[maxCastDataCount], dataCount);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Muls(float32Tensor, float32Tensor, scalarVal, dataCount);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Cast(tensor2Local, float32Tensor, AscendC::RoundMode::CAST_RINT, dataCount);
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Axpy<T, T>(tensor1Local, tensor2Local, (T)1, dataCount);  // //?: 标量已并入 Muls，Axpy 系数取 1 是否最优待销案
    }
}

// ---- 工厂子类（按 R17/R18 学到的钩子形态自写主循环复述版）----
template <typename T, int32_t bufferNum = BUFFER_NUM, uint8_t paramsCount = 3>
class MyForeachAddcdivScalar {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR scalar, GM_ADDR y,
                                GM_ADDR workspace, const ForeachCommonTilingData* tilingData)
    {
        this->td = tilingData;
        inPtr1 = x1; inPtr2 = x2; inPtr3 = x3; outPtr = y;
        blockIdx = AscendC::GetBlockIdx();
        scalarGM.SetGlobalBuffer((__gm__ T*)scalar, 1);
        // 标量读入（B10：MTE2→S 事件对）
        pipe.InitBuffer(scalarBuf, BYTE_BLOCK);
        AscendC::LocalTensor<T> sLocal = scalarBuf.Get<T>();
        AscendC::DataCopy(sLocal, scalarGM, BYTE_BLOCK / sizeof(T));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        scalarVal = (float)sLocal.GetValue(0);

        pipe.InitBuffer(dataQueue1, bufferNum, td->inputsTensorUbSize);
        pipe.InitBuffer(inQueue2, bufferNum, td->inputsTensorUbSize);
        pipe.InitBuffer(inQueue3, bufferNum, td->inputsTensorUbSize);
        maxDataCount = td->inputsTensorUbSize / sizeof(T);
        if constexpr (!std::is_same_v<T, float>) {
            pipe.InitBuffer(f32Queue, 1, td->inputsTensorUbSize * paramsCount);  // //?: paramsCount 份 fp32 槽
        }
    }

    __aicore__ inline void Process()
    {
        for (uint16_t idx = td->tensorStartList[blockIdx]; idx <= td->tensorEndList[blockIdx]; idx++) {
            uint64_t cs = (idx == td->tensorStartList[blockIdx]) ? td->tensorStartOffsetList[blockIdx] : 0;
            uint64_t ce = (idx == td->tensorEndList[blockIdx]) ? td->tensorEndOffsetList[blockIdx]
                                                               : td->tensorDataCountList[idx];
            gm1.SetGlobalBuffer(GetAddr(idx, inPtr1) + cs);
            gm2.SetGlobalBuffer(GetAddr(idx, inPtr2) + cs);
            gm3.SetGlobalBuffer(GetAddr(idx, inPtr3) + cs);
            gmOut.SetGlobalBuffer(GetAddr(idx, outPtr) + cs);

            for (uint64_t cur = cs; cur < ce; cur += maxDataCount) {
                int64_t seg = (ce - cur) < maxDataCount ? (ce - cur) : maxDataCount;
                CopyIn(cur - cs, seg);
                Compute(seg);
                CopyOut(cur - cs, seg);   // 隐式输出的显式收尾：结果从 x1 缓冲写出
            }
        }
    }

private:
    __aicore__ inline __gm__ T* GetAddr(uint16_t index, GM_ADDR tensorPtr)
    {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        uint64_t off = *dataAddr;
        return reinterpret_cast<__gm__ T*>(*(dataAddr + (off >> 3) + index));
    }
    __aicore__ inline void CopyInSeg(AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>& q,
                                     AscendC::GlobalTensor<T>& gm, uint64_t off, int64_t count)
    {
        AscendC::LocalTensor<T> t = q.AllocTensor<T>();
        if (count == maxDataCount) {
            AscendC::DataCopy(t, gm[off], count);
        } else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(t, gm[off], cp, pad);
        }
        q.EnQue(t);
    }
    __aicore__ inline void CopyIn(uint64_t off, int64_t count)
    {
        CopyInSeg(dataQueue1, gm1, off, count);
        CopyInSeg(inQueue2, gm2, off, count);
        CopyInSeg(inQueue3, gm3, off, count);
    }
    __aicore__ inline void Compute(int64_t count)
    {
        AscendC::LocalTensor<T> t1 = dataQueue1.DeQue<T>();
        AscendC::LocalTensor<T> t2 = inQueue2.DeQue<T>();
        AscendC::LocalTensor<T> t3 = inQueue3.DeQue<T>();
        AscendC::LocalTensor<float> f32 = f32Queue.Get<float>();  // //?: fp32 槽的常驻/轮转方式
        MyAddcDivAdapter<T>(t1, t2, t3, f32, scalarVal, maxDataCount, count);
        dataQueue1.EnQue(t1);  // 结果在 t1 里，回队供 CopyOut（隐式输出）
        inQueue2.FreeTensor(t2);
        inQueue3.FreeTensor(t3);
    }
    __aicore__ inline void CopyOut(uint64_t off, int64_t count)
    {
        AscendC::LocalTensor<T> t1 = dataQueue1.DeQue<T>();
        if (count == maxDataCount) {
            AscendC::DataCopy(gmOut[off], t1, count);
        } else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(gmOut[off], t1, cp);
        }
        dataQueue1.FreeTensor(t1);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> dataQueue1, inQueue2, inQueue3;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> f32Queue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<T> gm1, gm2, gm3, gmOut, scalarGM;
    GM_ADDR inPtr1, inPtr2, inPtr3, outPtr;
    const ForeachCommonTilingData* td;
    float scalarVal;
    uint16_t blockIdx;
    int64_t maxDataCount;
};

extern "C" __global__ __aicore__ void foreach_addcdiv_scalar_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3,
                                                                    GM_ADDR scalar, GM_ADDR y,
                                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachCommonTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2)) {
        MyForeachAddcdivScalar<float> op;
        op.Init(x1, x2, x3, scalar, y, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        MyForeachAddcdivScalar<half> op;
        op.Init(x1, x2, x3, scalar, y, workspace, &tilingData);
        op.Process();
    }
}
