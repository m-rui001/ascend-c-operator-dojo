/*
 * Round 30 - 我实现的 foreach_copy：单队列直出（无 VECOUT 队列的退化流水）
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class MyForeachCopy {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* td)
    {
        this->td = td;
        blockIdx = AscendC::GetBlockIdx();
        inPtr = x; outPtr = y;
        pipe.InitBuffer(dataQueue, BUFFER_NUM, td->inputsTensorUbSize);
        maxDataCount = td->inputsTensorUbSize / sizeof(T);
    }

    __aicore__ inline void Process()
    {
        for (uint16_t idx = td->tensorStartList[blockIdx]; idx <= td->tensorEndList[blockIdx]; idx++) {
            uint64_t cs = (idx == td->tensorStartList[blockIdx]) ? td->tensorStartOffsetList[blockIdx] : 0;
            uint64_t ce = (idx == td->tensorEndList[blockIdx]) ? td->tensorEndOffsetList[blockIdx]
                                                               : td->tensorDataCountList[idx];
            inGM.SetGlobalBuffer(GetAddr(idx, inPtr) + cs);
            outGM.SetGlobalBuffer(GetAddr(idx, outPtr) + cs);
            for (uint64_t cur = cs; cur < ce; cur += maxDataCount) {
                int64_t seg = (ce - cur) < maxDataCount ? (ce - cur) : maxDataCount;
                CopyInAndCopyOut(cur - cs, seg);
            }
        }
    }

private:
    __aicore__ inline __gm__ T* GetAddr(uint16_t index, GM_ADDR tensorPtr)
    {
        __gm__ uint64_t* d = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        return reinterpret_cast<__gm__ T*>(*(d + ((*d) >> 3) + index));
    }
    __aicore__ inline void CopyInAndCopyOut(uint64_t off, int64_t count)
    {
        AscendC::LocalTensor<T> local = dataQueue.AllocTensor<T>();
        if (count == maxDataCount) {
            AscendC::DataCopy(local, inGM[off], count);
        } else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(local, inGM[off], cp, pad);
        }
        dataQueue.EnQue(local);
        AscendC::LocalTensor<T> in = dataQueue.DeQue<T>();
        if (count == maxDataCount) {
            AscendC::DataCopy(outGM[off], in, count);
        } else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(outGM[off], in, cp);
        }
        dataQueue.FreeTensor(in);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> dataQueue;
    AscendC::GlobalTensor<T> inGM, outGM;
    GM_ADDR inPtr, outPtr;
    const ForeachCommonTilingData* td;
    uint16_t blockIdx;
    int64_t maxDataCount;
};
