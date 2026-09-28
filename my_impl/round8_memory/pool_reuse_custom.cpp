/*
 * Round 8 - 我自己写的内存语义练习算子：两阶段 Add/Sub + TBufPool 池复用 + in-place + L2 hint
 * 阶段1（前半）：z1 = x1 + y1（in-place Add，dst 复用 xLocal 缓冲）
 * 阶段2（后半）：z2 = x2 - y2
 * pool2 与 pool1 复用同一块 UB；x/y/z 均为流过数据，L2 DISABLE
 * 对没把握的 API 用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;

class KernelPoolReuse {
public:
    __aicore__ inline KernelPoolReuse() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t stageLength)  // stageLength = 每阶段元素数
    {
        xGm.SetGlobalBuffer((__gm__ half*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ half*)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ half*)z, totalLength);
        this->stageLength = stageLength;

        // //?? #1: 池的创建与复用声明——API 形态是猜的
        // 假设：TPipe::InitBufPool(pool, 起始偏移/大小) 或先 InitBufPool 再由 pool.InitBuffer
        pipe.InitBufPool(tbufPool1, stageLength * sizeof(half));       // //??
        tbufPool1.InitBuffer(src0Buf, stageLength * sizeof(half));     // //?? pool 内分配语义
        tbufPool1.InitBuffer(src1Buf, stageLength * sizeof(half));

        // //?? #3: 复用声明方式：假设 pool2 与 pool1 共享地址的 API 是 SetBufPoolReuse 或构造参数
        tbufPool2.SetReuse(tbufPool1, stageLength * sizeof(half));     // //?? 纯猜测

        // L2 hint：三路数据都是一次流过 //?? #4: 枚举值是否准确
        xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        zGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    __aicore__ inline void Process()
    {
        Stage1Add(0);            // 前半：Add（in-place）
        Stage2Sub(stageLength);  // 后半：Sub，复用同一块 UB
    }

private:
    // 阶段1：z1 = x1 + y1，in-place（dst 与 src0 100% 重叠）
    __aicore__ inline void Stage1Add(uint32_t offset)
    {
        AscendC::LocalTensor<half> xLocal = src0Buf.Get<half>();
        AscendC::LocalTensor<half> yLocal = src1Buf.Get<half>();
        AscendC::DataCopy(xLocal, xGm[offset], stageLength);
        AscendC::DataCopy(yLocal, yGm[offset], stageLength);
        // //?? #5: in-place 条件：dst 与第二源重叠、stride=0，本场景应满足
        AscendC::Add<half>(xLocal, xLocal, yLocal, stageLength);
        // //?? #6: 串行场景直接用 TBuf 而非 TQue，无需 EnQue/DeQue
        AscendC::DataCopy(zGm[offset], xLocal, stageLength);
    }

    // 阶段2：z2 = x2 - y2，pool2 与 pool1 物理复用
    __aicore__ inline void Stage2Sub(uint32_t offset)
    {
        AscendC::LocalTensor<half> xLocal = src0Buf.Get<half>();  // //?? pool2 视角的 Get 是否同一地址
        AscendC::LocalTensor<half> yLocal = src1Buf.Get<half>();
        AscendC::DataCopy(xLocal, xGm[offset], stageLength);
        AscendC::DataCopy(yLocal, yGm[offset], stageLength);
        AscendC::Sub<half>(xLocal, xLocal, yLocal, stageLength);
        AscendC::DataCopy(zGm[offset], xLocal, stageLength);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBufPool<AscendC::QuePosition::VECCALC> tbufPool1, tbufPool2;  // //?? 模板参数猜测
    AscendC::TBuf<AscendC::QuePosition::VECCALC> src0Buf, src1Buf;
    AscendC::GlobalTensor<half> xGm, yGm, zGm;
    uint32_t stageLength;
};

extern "C" __global__ __aicore__ void pool_reuse_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelPoolReuse op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.stageLength);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void pool_reuse_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                          uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* workspace, uint8_t* tiling) {
    pool_reuse_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif
