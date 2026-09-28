/*
 * Round 3 - 我自己写的 ReduceSum 算子（last dim 归约，fold 前导维为行）
 * 本轮重点：API 选型（WholeReduceSum）、输出攒批写出、fp32 累加精度
 * 对 API 细节没把握处用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelReduceSum {
public:
    __aicore__ inline KernelReduceSum() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint64_t colLength,   // 每行 N
                                uint64_t smallRows, uint64_t tailRows,
                                uint64_t ubTile,      // 每片列数
                                uint64_t colTail)     // 尾片列数
    {
        uint64_t blockIdx = AscendC::GetBlockIdx();
        rows = smallRows + (blockIdx < tailRows ? 1 : 0);
        uint64_t rowOffset = blockIdx * smallRows + (blockIdx < tailRows ? blockIdx : tailRows);
        this->cols = colLength;
        this->ubTile = ubTile;
        this->colTail = colTail;
        this->tileNum = (colLength + ubTile - 1) / ubTile;

        xGm.SetGlobalBuffer((__gm__ T*)x + rowOffset * colLength, rows * colLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + rowOffset, rows);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubTile * sizeof(T));
        // fp32 累加用临时片（行内跨片求和的载体）
        pipe.InitBuffer(accumBuf, 32 * sizeof(float));  //?? 归约中间量到底要多大
        // 输出攒批：本核所有行结果先攒在这，最后一次性写出
        pipe.InitBuffer(outBuf, rows * sizeof(T) < 32 ? 32 : rows * sizeof(T));  //?? 未 32B 向上取整
    }

    __aicore__ inline void Process() {
        AscendC::LocalTensor<T> yLocal = outBuf.Get<T>();
        for (uint64_t r = 0; r < rows; r++) {
            float rowSum = 0.0f;
            AscendC::LocalTensor<float> accum = accumBuf.Get<float>();
            for (uint64_t t = 0; t < tileNum; t++) {
                uint32_t count = (t == tileNum - 1) ? (uint32_t)colTail : (uint32_t)ubTile;
                AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
                AscendC::DataCopy(xLocal, xGm[r * cols + t * ubTile], count);
                inQueueX.EnQue(xLocal);
                AscendC::LocalTensor<T> xIn = inQueueX.DeQue<T>();
                //?? WholeReduceSum 签名：dst/fp32 每行一个值？先按"归约到 accum[0]"假设
                AscendC::WholeReduceSum<T, float>(accum, xIn, count);
                rowSum += accum.GetValue(0);
                inQueueX.FreeTensor(xIn);
            }
            yLocal.SetValue(r, (T)rowSum);  //?? SetValue 是标量写 UB 的正确方式吗
        }
        // 攒批写出（//?? rows*sizeof(T) 不是 32B 倍数时 DataCopy 会不会越界）
        AscendC::DataCopy(yGm, yLocal, rows);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accumBuf, outBuf;
    AscendC::GlobalTensor<T> xGm, yGm;
    uint64_t rows, cols, ubTile, colTail, tileNum;
};

extern "C" __global__ __aicore__ void reduce_sum_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelReduceSum<DTYPE_X> op;
    op.Init(x, y, tiling_data.colLength, tiling_data.smallRows, tiling_data.tailRows,
            tiling_data.ubTile, tiling_data.colTail);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void reduce_sum_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                          uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling) {
    reduce_sum_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
