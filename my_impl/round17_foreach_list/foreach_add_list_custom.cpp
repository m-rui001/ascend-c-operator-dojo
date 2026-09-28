/*
 * Round 17 - 我预测式实现的 foreach_add_list：out[i] = x1[i] + alpha * x2[i]
 * 按我理解的"算子工厂"模型写：模板方法钩子 + 第二队列 + 逐张量地址重绑 + 尾段 pad
 * 与真实 ForeachOneScalarTernary 对比后销案（见复盘）
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

// 我的函数对象签名预测：OneScalarTernaryOp 形态
struct MyAddListOp {
    template <typename T>
    __aicore__ inline void operator()(LocalTensor<T>& out, const LocalTensor<T>& in1,
                                     const LocalTensor<T>& in2, T alpha, int64_t count) const {
        AscendC::Muls(out, in2, alpha, count);   // //?: 次序——先乘 alpha 还是先加
        AscendC::Add(out, in1, out, count);      // in-place Add（dst=src1 重叠，CHECKLIST B5）
    }
};

template <typename T, typename OP, int32_t bufferNum = BUFFER_NUM>
class MyForeachAddList {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha, GM_ADDR y,
                                GM_ADDR workspace, const ForeachCommonTilingData* tilingData)
    {
        this->tilingData = tilingData;
        inTensorsPtr_1 = x1;
        inTensorsPtr_2 = x2;
        outTensorsPtr = y;
        inScalarGM.SetGlobalBuffer((__gm__ T*)alpha, 1);
        alphaVal = inScalarGM.GetValue(0);  // //?: alpha 每核读一次 vs 每张量读一次（标量列表时必须逐张量）
        blockIdx = AscendC::GetBlockIdx();

        pipe.InitBuffer(dataQueue, bufferNum, tilingData->maxDataCount * sizeof(T));
        pipe.InitBuffer(inQueue2, bufferNum, tilingData->maxDataCount * sizeof(T));
        pipe.InitBuffer(outQueue, bufferNum, tilingData->maxDataCount * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 按我理解的基类主循环：遍历本核 [tensorStart, tensorEnd) 的张量，每张量按 maxDataCount 分段
        for (uint16_t idx = tilingData->tensorStartList[blockIdx];
             idx <= tilingData->tensorEndList[blockIdx]; idx++) {
            uint64_t cursorStart = (idx == tilingData->tensorStartList[blockIdx])
                                       ? tilingData->tensorStartOffsetList[blockIdx] : 0;
            uint64_t cursorEnd = (idx == tilingData->tensorEndList[blockIdx])
                                     ? tilingData->tensorEndOffsetList[blockIdx]
                                     : tilingData->tensorDataCountList[idx];
            // 逐张量重绑三路地址
            inTensorsGM_1.SetGlobalBuffer(GetTensorAddr(idx, inTensorsPtr_1) + cursorStart);
            inTensorsGM_2.SetGlobalBuffer(GetTensorAddr(idx, inTensorsPtr_2) + cursorStart);
            outTensorsGM.SetGlobalBuffer(GetTensorAddr(idx, outTensorsPtr) + cursorStart);

            uint64_t cursor = cursorStart;
            while (cursor < cursorEnd) {
                int64_t seg = (cursorEnd - cursor) < tilingData->maxDataCount ? (cursorEnd - cursor)
                                                                              : tilingData->maxDataCount;
                CopyIn(cursor - cursorStart, seg);
                Compute(seg);
                CopyOut(cursor - cursorStart, seg);
                cursor += tilingData->maxDataCount;
            }
        }
    }

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
    {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        uint64_t tensorPtrOffset = *dataAddr;
        __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T*>(*(retPtr + index));
    }

    __aicore__ inline void CopyIn(uint64_t inTensorOff, int64_t count)
    {
        LocalTensor<T> in1 = dataQueue.AllocTensor<T>();
        CopySeg(in1, inTensorsGM_1, inTensorOff, count);
        dataQueue.EnQue(in1);
        LocalTensor<T> in2 = inQueue2.AllocTensor<T>();
        CopySeg(in2, inTensorsGM_2, inTensorOff, count);
        inQueue2.EnQue(in2);
    }

    __aicore__ inline void CopySeg(LocalTensor<T>& dst, AscendC::GlobalTensor<T>& gm,
                                   uint64_t off, int64_t count)
    {
        if (count == tilingData->maxDataCount) {
            AscendC::DataCopy(dst, gm[off], count);
        } else {  // 尾段：字节粒度 pad
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(dst, gm[off], cp, pad);
        }
    }

    __aicore__ inline void Compute(int64_t count)
    {
        LocalTensor<T> in1 = dataQueue.DeQue<T>();
        LocalTensor<T> in2 = inQueue2.DeQue<T>();
        LocalTensor<T> out = outQueue.AllocTensor<T>();
        MyAddListOp()(out, in1, in2, alphaVal, count);
        outQueue.EnQue(out);
        dataQueue.FreeTensor(in1);
        inQueue2.FreeTensor(in2);
    }

    __aicore__ inline void CopyOut(uint64_t off, int64_t count)
    {
        LocalTensor<T> out = outQueue.DeQue<T>();
        if (count == tilingData->maxDataCount) {
            AscendC::DataCopy(outTensorsGM[off], out, count);
        } else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(outTensorsGM[off], out, cp);
        }
        outQueue.FreeTensor(out);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, bufferNum> dataQueue, inQueue2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, bufferNum> outQueue;
    AscendC::GlobalTensor<T> inTensorsGM_1, inTensorsGM_2, outTensorsGM, inScalarGM;
    GM_ADDR inTensorsPtr_1, inTensorsPtr_2, outTensorsPtr;
    const ForeachCommonTilingData* tilingData;
    T alphaVal;
    uint16_t blockIdx;
};

extern "C" __global__ __aicore__ void foreach_add_list_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha,
                                                              GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2)) {
        MyForeachAddList<float, MyAddListOp> op;
        op.Init(x1, x2, alpha, y, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        MyForeachAddList<half, MyAddListOp> op;
        op.Init(x1, x2, alpha, y, workspace, &tilingData);
        op.Process();
    }
}
