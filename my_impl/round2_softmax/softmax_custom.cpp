/*
 * Round 2 - 我自己写的 Softmax 算子（SIMD/SPMD，按行 softmax，last dim 归约）
 * 应用 Round 1 教训：uint64、大小核行切分、尾块、模板 dtype、TBuf 中间量
 * 写作时未参考社区 softmax 实现；对 API 细节没把握处用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint64_t colLength,           // 每行列数 N
                                uint64_t smallRows,           // 小核行数
                                uint64_t tailRows,            // 前 tailRows 个核每核多一行
                                uint64_t ubTile,              // 每片列数（UB 容量反推）
                                uint64_t colTail)             // 尾片列数
    {
        // 大小核行切分：前 tailRows 个核各多一行（Round 1 教训：不假设整除）
        uint64_t blockIdx = AscendC::GetBlockIdx();
        uint64_t rowsPerCore = smallRows + (blockIdx < tailRows ? 1 : 0);
        uint64_t rowOffset = blockIdx * smallRows + (blockIdx < tailRows ? blockIdx : tailRows);
        this->rows = rowsPerCore;
        this->cols = colLength;
        this->ubTile = ubTile;
        this->colTail = colTail;
        this->tileNum = (colLength + ubTile - 1) / ubTile;  // 每行片数

        xGm.SetGlobalBuffer((__gm__ T*)x + rowOffset * colLength, rowsPerCore * colLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + rowOffset * colLength, rowsPerCore * colLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubTile * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ubTile * sizeof(T));
        // 中间量不进队列，用 TBuf：sub/exp 结果一片
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, ubTile * sizeof(T));  //?? TBuf 是否支持 BUFFER_NUM？
        // 行状态：1 元素足够，但按 32B 对齐申请
        pipe.InitBuffer(maxBuf, 1, 32);   //?? 标量状态到底放哪
        pipe.InitBuffer(sumBuf, 1, 32);
    }

    __aicore__ inline void Process() {
        for (uint64_t r = 0; r < rows; r++) {
            ComputeRow(r);
        }
    }

private:
    // 一次搬入一片（自动处理尾片）
    __aicore__ inline void CopyTileIn(uint64_t colOffset, uint32_t count) {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        //?? 尾片 count 不足 32B 对齐时，DataCopy 是否安全（Round1 遗留问题）
        AscendC::DataCopy(xLocal, xGm[r * cols + colOffset], count);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputeRow(uint64_t r) {
        this->r = r;
        // ---- Pass 1: 行 max ----
        float rowMax = -3.4e38f;
        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = (t == tileNum - 1) ? (uint32_t)colTail : (uint32_t)ubTile;
            CopyTileIn(t * ubTile, count);
            AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            AscendC::LocalTensor<T> maxT = maxBuf.Get<T>();
            //?? ReduceMax 的准确签名/输出语义：假设归约结果放在 dst[0]
            AscendC::ReduceMax<T>(maxT, xLocal, workLocal, count);
            float tileMax = (float)maxT.GetValue(0);
            if (tileMax > rowMax) rowMax = tileMax;
            inQueueX.FreeTensor(xLocal);
        }

        // ---- Pass 2: exp 并累加行 sum ----
        float rowSum = 0.0f;
        AscendC::LocalTensor<T> sumT = sumBuf.Get<T>();
        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = (t == tileNum - 1) ? (uint32_t)colTail : (uint32_t)ubTile;
            CopyTileIn(t * ubTile, count);
            AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            AscendC::LocalTensor<T> sub = tmpBuf.Get<T>(0);
            AscendC::LocalTensor<T> exp = tmpBuf.Get<T>(1);
            AscendC::Adds<T>(sub, xLocal, (T)(-rowMax), count);
            AscendC::Exp(exp, sub, count);
            AscendC::ReduceSum<T>(sumT, exp, workLocal, count);  //?? 同上，输出 sumT[0]
            rowSum += (float)sumT.GetValue(0);
            inQueueX.FreeTensor(xLocal);
        }

        // ---- Pass 3: 重算 exp 并除以 sum 写出 ----
        float invSum = 1.0f / rowSum;
        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = (t == tileNum - 1) ? (uint32_t)colTail : (uint32_t)ubTile;
            CopyTileIn(t * ubTile, count);
            AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            AscendC::LocalTensor<T> sub = tmpBuf.Get<T>(0);
            AscendC::LocalTensor<T> exp = tmpBuf.Get<T>(1);
            AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
            AscendC::Adds<T>(sub, xLocal, (T)(-rowMax), count);
            AscendC::Exp(exp, sub, count);
            AscendC::Muls(exp, exp, invSum, count);  // 用倒数乘法代替除法 //?? 精度损失可接受？
            AscendC::DataCopy(yLocal, exp, count);   //?? 同 dtype 拷贝还是直接让 yLocal=exp？
            outQueueY.EnQue(yLocal);
            AscendC::LocalTensor<T> yOut = outQueueY.DeQue<T>();
            AscendC::DataCopy(yGm[r * cols + t * ubTile], yOut, count);
            outQueueY.FreeTensor(yOut);
            inQueueX.FreeTensor(xLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maxBuf, sumBuf;
    AscendC::GlobalTensor<T> xGm, yGm;
    AscendC::LocalTensor<T> workLocal;  //?? ReduceMax/ReduceSum 需要 workspace？我猜的
    uint64_t rows, cols, ubTile, colTail, tileNum, r;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax<DTYPE_X> op;
    op.Init(x, y, tiling_data.colLength,
            tiling_data.smallRows, tiling_data.tailRows,
            tiling_data.ubTile, tiling_data.colTail);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                       uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling) {
    softmax_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
