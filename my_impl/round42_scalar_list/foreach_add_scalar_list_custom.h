/*
 * Round 42 - 我实现的 foreach_add_scalar_list：标量列表 + ProcessPlusInLoop 钩子验证
 * 要点：标量列表直接 GM GetValue（免 DataCopy 搬运）；钩子在每张量处理前刷新标量
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;  // 与生产实例化一致（1, 1）

// ---- 基类模拟：主循环 + 钩子调用点（生产在 KernelForeachUnary 中）----
template <typename T>
class MyScalarListBase {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scalarList, GM_ADDR y,
                                const ForeachCommonTilingData* td)
    {
        this->td = td;
        blockIdx = AscendC::GetBlockIdx();
        scalarGM.SetGlobalBuffer((__gm__ T*)scalarList, td->tensorDataCountList[td->totalTensorCount - 1] > 0
                                                         ? MAX_SCALARS : MAX_SCALARS);
        pipe.InitBuffer(dataQueue, BUFFER_NUM, td->inputsTensorUbSize);
        maxDataCount = td->inputsTensorUbSize / sizeof(T);
    }

    __aicore__ inline void Process()
    {
        for (uint16_t idx = td->tensorStartList[blockIdx]; idx <= td->tensorEndList[blockIdx]; idx++) {
            uint64_t cs = (idx == td->tensorStartList[blockIdx]) ? td->tensorStartOffsetList[blockIdx] : 0;
            uint64_t ce = (idx == td->tensorEndList[blockIdx]) ? td->tensorEndOffsetList[blockIdx]
                                                               : td->tensorDataCountList[idx];
            // 钩子调用点：每张量处理前（生产同位）
            ProcessPlusInLoop(idx, cs);
            for (uint64_t cur = cs; cur < ce; cur += maxDataCount) {
                int64_t seg = (ce - cur) < maxDataCount ? (ce - cur) : maxDataCount;
                CopyIn(idx, cur - cs, seg);
                Compute(idx, seg, false);
                CopyOut(idx, cur - cs, seg);
            }
        }
    }

protected:
    // 子类钩子
    __aicore__ inline void Compute(uint32_t index, int64_t count, bool isRemainder)
    {
        AscendC::LocalTensor<T> in = dataQueue.DeQue<T>();
        AscendC::Adds(in, in, scalarVal, count);   // 隐式输出：原地
        // CopyOut 由基类做（本模拟合并）
        AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopyPad(outGM[index * maxDataCount], in, cp);
        dataQueue.FreeTensor(in);
    }
    // 标量列表钩子：每张量刷新标量（本轮验证点）
    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart)
    {
        scalarVal = scalarGM.GetValue(index);   // GM 直接读，O(张量数) 允许级
    }

private:
    __aicore__ inline __gm__ T* GetAddr(uint16_t index, GM_ADDR tensorPtr)
    {
        __gm__ uint64_t* d = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        return reinterpret_cast<__gm__ T*>(*(d + ((*d) >> 3) + index));
    }
    __aicore__ inline void CopyIn(uint32_t index, uint64_t off, int64_t count)
    {
        inGM.SetGlobalBuffer(GetAddr(index, inPtr) + off);
        AscendC::LocalTensor<T> t = dataQueue.AllocTensor<T>();
        if (count == maxDataCount) {
            AscendC::DataCopy(t, inGM[0], count);
        } else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(t, inGM[0], cp, pad);
        }
        dataQueue.EnQue(t);
    }
    __aicore__ inline void CopyOut(uint32_t index, uint64_t off, int64_t count)
    {
        outGM.SetGlobalBuffer(GetAddr(index, outPtr) + off);
    }

protected:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> dataQueue;
    AscendC::GlobalTensor<T> inGM, outGM, scalarGM;
    GM_ADDR inPtr, outPtr;
    const ForeachCommonTilingData* td;
    T scalarVal;
    uint16_t blockIdx;
    int64_t maxDataCount;
    static constexpr uint32_t MAX_SCALARS = 64;
};

template <typename T>
class MyForeachAddScalarList : public MyScalarListBase<T> {
public:
    __aicore__ inline void InitAll(GM_ADDR x, GM_ADDR scalarList, GM_ADDR y, const ForeachCommonTilingData* td)
    {
        this->Init(x, scalarList, y, td);
        this->inPtr = x;
        this->outPtr = y;
    }
};

extern "C" __global__ __aicore__ void foreach_add_scalar_list_custom(GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
                                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachCommonTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        MyForeachAddScalarList<half> op;
        op.InitAll(x, scalar, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        MyForeachAddScalarList<float> op;
        op.InitAll(x, scalar, y, &tilingData);
        op.Process();
    }
}
