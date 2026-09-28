/*
 * Round 16 - 我自己写的 foreach_add_scalar：张量列表 + 标量，逐张量 Adds
 * 结构新点：变长张量列表的描述表传递、跨张量多核分配、Device 侧标量
 * 对没把握处用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_TENSORS = 64;  // //?: 描述表定长上限的取舍待销案

struct TensorDesc {   // //?: 描述表 entry 的载体与存放位置待销案
    uint64_t gmAddr;      // 张量设备地址（host 侧取得）
    uint64_t elemCount;
};

template <typename T>
class KernelForeachAddScalar {
public:
    __aicore__ inline void Init(GM_ADDR scalar, GM_ADDR descTable, GM_ADDR outList,
                                uint32_t tensorNum, uint32_t ubTile, float scalarFallback)
    {
        this->tensorNum = tensorNum;
        this->ubTile = ubTile;
        // 标量在 Device GM 上：先读进 UB 再取标量（CHECKLIST B10：V_S 事件对）
        scalarGm.SetGlobalBuffer((__gm__ T*)scalar, 1);
        pipe.InitBuffer(scalarBuf, BLOCK_SIZE);
        AscendC::LocalTensor<T> sLocal = scalarBuf.Get<T>();
        // //?: 单元素 DataCopy 不足 32B——按整 32B 块拷（对齐读取）
        AscendC::DataCopy(sLocal, scalarGm, BLOCK_SIZE / sizeof(T));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);   // //?: 事件对方向核查
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        this->scalarValue = sLocal.GetValue(0);

        // 描述表从 GM 读入并解析 //?: 是否有更优的一次性批量读法
        descGm.SetGlobalBuffer((__gm__ TensorDesc*)descTable, tensorNum);
        pipe.InitBuffer(descBuf, tensorNum * sizeof(TensorDesc) < BLOCK_SIZE ? BLOCK_SIZE : tensorNum * sizeof(TensorDesc));
        AscendC::LocalTensor<TensorDesc> dLocal = descBuf.Get<TensorDesc>();
        AscendC::DataCopy(dLocal, descGm, tensorNum * sizeof(TensorDesc));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);

        // 贪心块分配：全局块序 → 本核区间
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        uint64_t totalBlocks = 0;
        for (uint32_t i = 0; i < tensorNum; i++) {
            totalBlocks += (dLocal(i).elemCount + ubTile - 1) / ubTile;  // //?: dLocal(i) 取法待销案
        }
        uint64_t blocksPerCore = (totalBlocks + blockNum - 1) / blockNum;
        myStart = blockIdx * blocksPerCore;
        myEnd = myStart + blocksPerCore < totalBlocks ? myStart + blocksPerCore : totalBlocks;

        pipe.InitBuffer(inQueue, BUFFER_NUM, ubTile * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, ubTile * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 顺序扫过本核区间，locate 张量归属
        uint64_t blockIdxRun = 0;
        for (uint32_t i = 0; i < tensorNum; i++) {
            TensorDesc d = GetDesc(i);  // //?: 标量通路批量读后的逐个访问方式
            uint64_t tensorBlocks = (d.elemCount + ubTile - 1) / ubTile;
            for (uint64_t b = 0; b < tensorBlocks; b++) {
                if (blockIdxRun + b < myStart || blockIdxRun + b >= myEnd) continue;
                uint64_t off = b * ubTile;
                uint32_t count = (off + ubTile <= d.elemCount) ? ubTile : (uint32_t)(d.elemCount - off);
                ProcessBlock((__gm__ T*)d.gmAddr, off, count);
            }
            blockIdxRun += tensorBlocks;
        }
    }

private:
    __aicore__ inline void ProcessBlock(__gm__ T* tensorAddr, uint64_t off, uint32_t count)
    {
        xGm.SetGlobalBuffer(tensorAddr + off, count);
        oGm.SetGlobalBuffer(tensorAddr + off, count);  // foreach_add_scalar 是 out-of-place，out 描述表应有独立地址 //?: 简化
        AscendC::LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm, count);  // //?: 尾块不足 32B 需 DataCopyPad
        inQueue.EnQue(xLocal);
        AscendC::LocalTensor<T> xIn = inQueue.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        AscendC::Adds(yLocal, xIn, scalarValue, count);
        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xIn);
        AscendC::LocalTensor<T> yOut = outQueue.DeQue<T>();
        AscendC::DataCopy(oGm, yOut, count);
        outQueue.FreeTensor(yOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalarBuf, descBuf;
    AscendC::GlobalTensor<T> xGm, oGm, scalarGm;
    AscendC::GlobalTensor<TensorDesc> descGm;
    T scalarValue;
    uint32_t tensorNum, ubTile;
    uint64_t myStart, myEnd;
};

extern "C" __global__ __aicore__ void foreach_add_scalar_custom(GM_ADDR xList, GM_ADDR scalar, GM_ADDR outList,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {   // fp16
        KernelForeachAddScalar<half> op;
        op.Init(scalar, tiling_data.descTableAddr, outList, tiling_data.tensorNum, tiling_data.ubTile, 0.0f);
        op.Process();
    } else if (TILING_KEY_IS(2)) {  // fp32
        KernelForeachAddScalar<float> op;
        op.Init(scalar, tiling_data.descTableAddr, outList, tiling_data.tensorNum, tiling_data.ubTile, 0.0f);
        op.Process();
    }
}
