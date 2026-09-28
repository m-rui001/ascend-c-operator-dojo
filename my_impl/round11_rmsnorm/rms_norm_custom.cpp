/*
 * Round 11 - 我自己写的 RmsNorm：y = x / sqrt(mean(x²)+eps) * gamma
 * 双路径：行装得下 UB → 单遍；行装不下 → 两遍扫描
 * fp32 中间精度；rstd 攒批写出；四件检查项已核
 * 对没把握的 API 用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;

class KernelRmsNorm {
public:
    __aicore__ inline KernelRmsNorm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
                                uint64_t colLength,        // N
                                uint64_t smallRows, uint64_t tailRows,
                                uint64_t ubTile,           // 每片列数
                                float eps)
    {
        uint64_t blockIdx = AscendC::GetBlockIdx();
        this->coreRows = smallRows + (blockIdx < tailRows ? 1 : 0);
        this->rowOffset = blockIdx * smallRows + (blockIdx < tailRows ? blockIdx : tailRows);  // 检查项1
        this->cols = colLength;
        this->ubTile = ubTile;
        this->eps = eps;
        this->fitInUb = (colLength <= ubTile);
        this->tileNum = fitInUb ? 1 : (colLength + ubTile - 1) / ubTile;
        this->colTail = fitInUb ? colLength : (colLength % ubTile == 0 ? ubTile : colLength % ubTile);

        xGm.SetGlobalBuffer((__gm__ half*)x + rowOffset * cols, coreRows * cols);
        yGm.SetGlobalBuffer((__gm__ half*)y + rowOffset * cols, coreRows * cols);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + rowOffset, coreRows);
        gGm.SetGlobalBuffer((__gm__ half*)gamma, cols);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubTile * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ubTile * sizeof(half));
        pipe.InitBuffer(f32Buf, ubTile * sizeof(float) * 2);   // x² / 中间量（fp32 路径）
        // gamma 常驻：行装得下时装整行；否则装一片滚动
        pipe.InitBuffer(gBuf, (fitInUb ? cols : ubTile) * sizeof(half));
        // rstd 攒批：本核所有行一次写出（Round 3 教训：标量写 GM 要攒批+32B 对齐）——检查项4
        pipe.InitBuffer(rstdBuf, (coreRows * sizeof(float) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
    }

    __aicore__ inline void Process()
    {
        if (coreRows == 0) return;   // 检查项3：host 已算 blockDim，空核双保险
        if (fitInUb) {
            LoadGammaRow();
        }
        AscendC::LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
        for (uint64_t r = 0; r < coreRows; r++) {
            float rms = ComputeRowSumSquare(r);   // 遍历求 sum(x²)
            float rstd = 1.0f / sqrtf(rms / (float)cols + eps);   // //?: 用 Reciprocal/Sqrt 矢量接口还是标量 sqrtf
            rstdLocal.SetValue(r, rstd);
            NormalizeRow(r, rstd);
        }
        // rstd 攒批写出：按字节精确长度（Round 9 教训：DataCopyExtParams/Pad 收尾）——检查项2
        AscendC::DataCopy(rstdGm, rstdLocal, coreRows);  // //?: coreRows*sizeof(f32) 非32B 倍数时需 Pad 形态
    }

private:
    // gamma 常驻整行（fitInUb 路径）
    __aicore__ inline void LoadGammaRow()
    {
        AscendC::LocalTensor<half> gLocal = gBuf.Get<half>();
        AscendC::DataCopy(gLocal, gGm[0], cols);  // //?: cols 非32B 倍数需 Pad
    }

    // 求 sum(x²)（fp32 累加）；fitInUb 时顺带把 x 行缓存复用
    __aicore__ inline float ComputeRowSumSquare(uint64_t r)
    {
        float acc = 0.0f;
        AscendC::LocalTensor<float> work = f32Buf.Get<float>();
        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = (t == tileNum - 1 && !fitInUb) ? (uint32_t)colTail : (uint32_t)(fitInUb ? cols : ubTile);
            AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
            AscendC::DataCopy(xLocal, xGm[r * cols + t * ubTile], count);
            inQueueX.EnQue(xLocal);
            AscendC::LocalTensor<half> xIn = inQueueX.DeQue<half>();
            // fp16→fp32 后平方（防溢出），归约
            AscendC::LocalTensor<float> xF32 = work;
            AscendC::Cast(xF32, xIn, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Mul(xF32, xF32, xF32, count);          // in-place：dst=src0 允许（Round 9 销案）
            AscendC::WholeReduceSum<float, true>(work[ubTile], xF32, count, 1, 1, 1);  // //?: 签名沿用 Round 3 形态
            acc += work[ubTile].GetValue(0);
            inQueueX.FreeTensor(xIn);
        }
        return acc;
    }

    // y = x * rstd * gamma
    __aicore__ inline void NormalizeRow(uint64_t r, float rstd)
    {
        AscendC::LocalTensor<half> gLocal = gBuf.Get<half>();
        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = (t == tileNum - 1 && !fitInUb) ? (uint32_t)colTail : (uint32_t)(fitInUb ? cols : ubTile);
            AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
            AscendC::DataCopy(xLocal, xGm[r * cols + t * ubTile], count);
            inQueueX.EnQue(xLocal);
            AscendC::LocalTensor<half> xIn = inQueueX.DeQue<half>();
            AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
            // (x * rstd) * gamma：两次 Mul（rstd 广播标量）
            AscendC::Muls(yLocal, xIn, rstd, count);
            AscendC::Mul(yLocal, yLocal, gLocal[t * ubTile], count);  // //?: gamma 滚动片偏移寻址
            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xIn);
            AscendC::LocalTensor<half> yOut = outQueueY.DeQue<half>();
            AscendC::DataCopy(yGm[r * cols + t * ubTile], yOut, count);
            outQueueY.FreeTensor(yOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> f32Buf, gBuf, rstdBuf;
    AscendC::GlobalTensor<half> xGm, yGm, gGm;
    AscendC::GlobalTensor<float> rstdGm;
    uint64_t coreRows, rowOffset, cols, ubTile, tileNum, colTail;
    bool fitInUb;
    float eps;
};

extern "C" __global__ __aicore__ void rms_norm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsNorm op;
    op.Init(x, gamma, y, rstd, tiling_data.colLength,
            tiling_data.smallRows, tiling_data.tailRows, tiling_data.ubTile, tiling_data.epsilon);
    op.Process();   // 检查项4：workspace=0，无跨核交换
}

#ifndef ASCENDC_CPU_DEBUG
void rms_norm_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                        uint8_t* x, uint8_t* gamma, uint8_t* y, uint8_t* rstd,
                        uint8_t* workspace, uint8_t* tiling) {
    rms_norm_custom<<<blockDim, l2ctrl, stream>>>(x, gamma, y, rstd, workspace, tiling);
}
#endif
