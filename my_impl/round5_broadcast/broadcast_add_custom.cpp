/*
 * Round 5 - 我自己写的 Broadcast 二元算子（以 Add 为例，fold 为 (M,N) 两轴）
 * 三种模式：无广播 / 行广播(y 一行常驻 UB) / 列广播(y 标量通路 Adds)
 * 对 API 细节没把握处用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;

enum BroadcastMode : uint8_t { MODE_NONE = 0, MODE_ROW_BCAST = 1, MODE_COL_BCAST = 2 };

template <typename T>
class KernelBroadcastAdd {
public:
    __aicore__ inline KernelBroadcastAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint64_t rows, uint64_t cols,
                                uint8_t mode,
                                uint64_t smallRows, uint64_t tailRows,
                                uint64_t ubTile, uint64_t colTail)
    {
        uint64_t blockIdx = AscendC::GetBlockIdx();
        this->coreRows = smallRows + (blockIdx < tailRows ? 1 : 0);
        this->rowOffset = blockIdx * smallRows + (blockIdx < tailRows ? blockIdx : tailRows);
        this->rows = rows;
        this->cols = cols;
        this->mode = mode;
        this->ubTile = ubTile;
        this->tileNum = (cols + ubTile - 1) / ubTile;
        this->colTail = cols % ubTile == 0 ? ubTile : cols % ubTile;

        xGm.SetGlobalBuffer((__gm__ T*)x + rowOffset * cols, coreRows * cols);
        zGm.SetGlobalBuffer((__gm__ T*)z + rowOffset * cols, coreRows * cols);
        // y 的视图：行广播时全量(M 不需要)，列广播时只需要本核行的 y 值
        yGm.SetGlobalBuffer((__gm__ T*)y, mode == MODE_COL_BCAST ? rows : cols);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubTile * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, ubTile * sizeof(T));
        // y 常驻区：行广播装整行；列广播只装 1 元素；无广播按行滚动复用
        pipe.InitBuffer(yBuf, mode == MODE_ROW_BCAST || mode == MODE_NONE ? cols * sizeof(T) : BLOCK_SIZE);
    }

    __aicore__ inline void Process() {
        if (coreRows == 0) return;
        if (mode == MODE_ROW_BCAST) {
            LoadYRow();  // 行广播：y 整行一次进 UB，所有行复用
        }
        for (uint64_t r = 0; r < coreRows; r++) {
            ProcessRow(r);
        }
    }

private:
    __aicore__ inline void LoadYRow() {
        AscendC::LocalTensor<T> yLocal = yBuf.Get<T>();
        // //?? cols 非 32B 对齐时这里需要 DataCopyPad（对比环节销案）
        AscendC::DataCopy(yLocal, yGm[0], cols);
    }

    __aicore__ inline void ProcessRow(uint64_t r) {
        AscendC::LocalTensor<T> yRow = yBuf.Get<T>();
        // 列广播/无广播：本行先把 y 拷进 UB
        // 列广播 y 本行只有 1 元素；无广播 y 本行整行
        if (mode == MODE_COL_BCAST) {
            AscendC::DataCopy(yRow, yGm[rowOffset + r], 1);  //?? 单元素 DataCopy 是否合法(不足32B)
        } else if (mode == MODE_NONE) {
            AscendC::DataCopy(yRow, yGm[r * cols], cols);  // y 与 x 同布局，随行搬入
        }

        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = (t == tileNum - 1) ? (uint32_t)colTail : (uint32_t)ubTile;
            AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
            AscendC::DataCopy(xLocal, xGm[r * cols + t * ubTile], count);
            inQueueX.EnQue(xLocal);
            AscendC::LocalTensor<T> xIn = inQueueX.DeQue<T>();
            AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
            if (mode == MODE_COL_BCAST) {
                float yScalar = (float)yRow.GetValue(0);  // 标量通路：本行 y 是单值
                AscendC::Adds<T>(zLocal, xIn, (T)yScalar, count);
            } else {
                // //?? yRow[t*ubTile] 切片寻址的正确写法待销案
                AscendC::Add<T>(zLocal, xIn, yRow[t * ubTile], count);
            }
            outQueueZ.EnQue(zLocal);
            inQueueX.FreeTensor(xIn);
            AscendC::LocalTensor<T> zOut = outQueueZ.DeQue<T>();
            AscendC::DataCopy(zGm[r * cols + t * ubTile], zOut, count);
            outQueueZ.FreeTensor(zOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> yBuf;
    AscendC::GlobalTensor<T> xGm, yGm, zGm;
    uint64_t rows, cols, coreRows, ubTile, tileNum, colTail, rowOffset = 0;
    uint8_t mode;
};

extern "C" __global__ __aicore__ void broadcast_add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelBroadcastAdd<DTYPE_X> op;
    op.Init(x, y, z, tiling_data.rows, tiling_data.cols,
            tiling_data.mode, tiling_data.smallRows, tiling_data.tailRows,
            tiling_data.ubTile, tiling_data.colTail);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void broadcast_add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                             uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* workspace, uint8_t* tiling) {
    broadcast_add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif
