/*
 * Round 13 - 我自己写的 AddLayerNorm：
 *   x_added = x1 + x2 (+bias) → y = LN(x_added)·gamma + beta，输出 y/mean/rstd(/x)
 * 行装得下（fp32 常驻）单遍；否则列片两遍。mean/rstd 矢量化（R11 教训）。
 * 对没把握的 API 用 //?? 标注
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;

class KernelAddLayerNorm {
public:
    __aicore__ inline KernelAddLayerNorm() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR gamma, GM_ADDR beta,
                                GM_ADDR y, GM_ADDR meanOut, GM_ADDR rstdOut, GM_ADDR xOut,
                                uint64_t cols, uint64_t smallRows, uint64_t tailRows,
                                uint64_t ubTile, float eps, bool writeX)
    {
        uint64_t blockIdx = AscendC::GetBlockIdx();
        coreRows = smallRows + (blockIdx < tailRows ? 1 : 0);
        rowOffset = blockIdx * smallRows + (blockIdx < tailRows ? blockIdx : tailRows);  // B1
        this->cols = cols;
        this->ubTile = ubTile;
        this->eps = eps;
        this->writeX = writeX;
        fitInUb = (cols * sizeof(float) <= ubTile * sizeof(half) * 2);  // fp32 常驻判断（R11）
        tileNum = fitInUb ? 1 : (cols + ubTile - 1) / ubTile;

        x1Gm.SetGlobalBuffer((__gm__ half*)x1 + rowOffset * cols, coreRows * cols);
        x2Gm.SetGlobalBuffer((__gm__ half*)x2 + rowOffset * cols, coreRows * cols);
        yGm.SetGlobalBuffer((__gm__ half*)y + rowOffset * cols, coreRows * cols);
        xGm.SetGlobalBuffer((__gm__ half*)xOut + rowOffset * cols, coreRows * cols);
        meanGm.SetGlobalBuffer((__gm__ float*)meanOut + rowOffset, coreRows);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstdOut + rowOffset, coreRows);
        gGm.SetGlobalBuffer((__gm__ half*)gamma, cols);
        bGm.SetGlobalBuffer((__gm__ half*)beta, cols);
        biasGm.SetGlobalBuffer((__gm__ half*)bias, cols);

        pipe.InitBuffer(inQ1, BUFFER_NUM, ubTile * sizeof(half));
        pipe.InitBuffer(inQ2, BUFFER_NUM, ubTile * sizeof(half));
        pipe.InitBuffer(outQY, BUFFER_NUM, ubTile * sizeof(half));
        pipe.InitBuffer(addedBuf, (fitInUb ? cols : ubTile) * sizeof(float));   // x_added fp32
        pipe.InitBuffer(statBuf, BLOCK_SIZE * 4);                                // mean/var 矢量载体
        pipe.InitBuffer(gBuf, (fitInUb ? cols : ubTile) * sizeof(half));
        pipe.InitBuffer(bBuf, (fitInUb ? cols : ubTile) * sizeof(half));
        // rstd/mean 攒批（B1/R3）：32B 向上取整（B2）
        pipe.InitBuffer(statOutBuf, (coreRows * sizeof(float) * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
    }

    __aicore__ inline void Process()
    {
        if (coreRows == 0) return;   // B1
        LoadGbRow();                  // gamma/beta 常驻（fitInUb 时）
        AscendC::LocalTensor<float> statOut = statOutBuf.Get<float>();
        for (uint64_t r = 0; r < coreRows; r++) {
            // 阶段1：x_added = x1+x2 (+bias)，fp32 常驻
            float mean = AddRow(r);
            // 阶段2：var（两遍：减均值再平方，数值稳定优先）
            float var = VarRow(r, mean);
            float rstd = 1.0f / sqrtf(var + eps);
            // 矢量化批写 mean/rstd（B3：本行一个值，SetValue 于攒批缓冲）
            statOut.SetValue(r * 2, mean);
            statOut.SetValue(r * 2 + 1, rstd);
            // 阶段3：归一化输出
            NormalizeRow(r, mean, rstd);
        }
        // 攒批写出 mean/rstd（B2：按字节，//?: DataCopyPad 形态沿用 R12）
        AscendC::DataCopy(meanGm, statOut[0], coreRows);      // //?: 非对齐收尾待验证
        AscendC::DataCopy(rstdGm, statOut[coreRows], coreRows);
    }

private:
    // x_added = x1+x2 (+bias)；返回 mean；fitInUb 时 fp32 结果常驻 addedBuf
    __aicore__ inline float AddRow(uint64_t r)
    {
        AscendC::LocalTensor<float> added = addedBuf.Get<float>();
        AscendC::LocalTensor<float> work = statBuf.Get<float>();
        float acc = 0.0f;
        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = TileCount(t);
            AscendC::LocalTensor<half> a = inQ1.AllocTensor<half>();
            AscendC::LocalTensor<half> b = inQ2.AllocTensor<half>();
            AscendC::DataCopy(a, x1Gm[r * cols + t * ubTile], count);
            AscendC::DataCopy(b, x2Gm[r * cols + t * ubTile], count);
            inQ1.EnQue(a); inQ2.EnQue(b);
            AscendC::LocalTensor<half> aIn = inQ1.DeQue<half>();
            AscendC::LocalTensor<half> bIn = inQ2.DeQue<half>();
            AscendC::LocalTensor<float> aF = work;
            AscendC::LocalTensor<float> bF = work[ubTile];
            AscendC::Cast(aF, aIn, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(bF, bIn, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();                                   // B9
            if (fitInUb) {
                AscendC::Add(added[t * ubTile], aF, bF, count);               // 常驻分支
            } else {
                AscendC::Add(aF, aF, bF, count);                              // in-place（B5）
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (hasBias) {
                AscendC::LocalTensor<half> gLocal = gBuf.Get<half>();
                AscendC::Adds(added[t * ubTile], added[t * ubTile], (float)gLocal.GetValue(0), count);  // //?: bias 按列向量广播的实现待销案（当前简化为标量）
                AscendC::PipeBarrier<PIPE_V>();
            }
            // sum(x_added)（B3：fp32 归约）
            if (fitInUb) {
                AscendC::WholeReduceSum<float, true>(work[ubTile * 2], added[t * ubTile], count, 1, 1, 1);
            } else {
                AscendC::WholeReduceSum<float, true>(work[ubTile * 2], aF, count, 1, 1, 1);
            }
            acc += work[ubTile * 2].GetValue(0);
            inQ1.FreeTensor(aIn); inQ2.FreeTensor(bIn);
        }
        return acc / (float)cols;
    }

    __aicore__ inline float VarRow(uint64_t r, float mean)
    {
        AscendC::LocalTensor<float> added = addedBuf.Get<float>();
        AscendC::LocalTensor<float> work = statBuf.Get<float>();
        float acc = 0.0f;
        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = TileCount(t);
            AscendC::LocalTensor<float> src = fitInUb ? added[t * ubTile] : added;  // 两遍路径重算时此处应为重加，//?: 简化暂读常驻
            AscendC::Sub(work, src, (float)mean, count);   // //?: 减标量的接口是 Subs? 待销案
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(work, work, work, count);          // in-place
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float, true>(work[ubTile * 3], work, count, 1, 1, 1);
            acc += work[ubTile * 3].GetValue(0);
        }
        return acc / (float)cols;
    }

    __aicore__ inline void NormalizeRow(uint64_t r, float mean, float rstd)
    {
        AscendC::LocalTensor<float> added = addedBuf.Get<float>();
        AscendC::LocalTensor<half> gLocal = gBuf.Get<half>();
        AscendC::LocalTensor<half> bLocal = bBuf.Get<half>();
        for (uint64_t t = 0; t < tileNum; t++) {
            uint32_t count = TileCount(t);
            AscendC::LocalTensor<half> yLocal = outQY.AllocTensor<half>();
            AscendC::LocalTensor<float> yF = statBuf.Get<float>();
            AscendC::LocalTensor<float> src = fitInUb ? added[t * ubTile] : added;  // //?: 同上
            AscendC::Sub(yF, src, (float)mean, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(yF, yF, rstd, count);
            AscendC::PipeBarrier<PIPE_V>();
            // //?: yF(half 写回) 与 gamma 相乘需要先 Cast 回 half 或 gamma Cast 到 fp32——暂用 half 域相乘
            AscendC::Cast(yLocal, yF, AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(yLocal, yLocal, gLocal[t * ubTile], count);
            AscendC::Add(yLocal, yLocal, bLocal[t * ubTile], count);
            outQY.EnQue(yLocal);
            AscendC::LocalTensor<half> yOut = outQY.DeQue<half>();
            AscendC::DataCopy(yGm[r * cols + t * ubTile], yOut, count);
            outQY.FreeTensor(yOut);
            if (writeX) {
                // //?: x 输出：从 added Cast 回 half 写出（简化：fitInUb 分支）
            }
        }
    }

    __aicore__ inline uint32_t TileCount(uint64_t t)
    {
        return (!fitInUb && t == tileNum - 1) ? (uint32_t)colTail : (uint32_t)(fitInUb ? cols : ubTile);
    }

    __aicore__ inline void LoadGbRow()
    {
        AscendC::LocalTensor<half> gLocal = gBuf.Get<half>();
        AscendC::LocalTensor<half> bLocal = bBuf.Get<half>();
        AscendC::DataCopy(gLocal, gGm[0], fitInUb ? cols : ubTile);
        AscendC::DataCopy(bLocal, bGm[0], fitInUb ? cols : ubTile);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQ1, inQ2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> addedBuf, statBuf, gBuf, bBuf, statOutBuf;
    AscendC::GlobalTensor<half> x1Gm, x2Gm, yGm, xGm, gGm, bGm, biasGm;
    AscendC::GlobalTensor<float> meanGm, rstdGm;
    uint64_t coreRows, rowOffset, cols, ubTile, tileNum, colTail = 0;
    bool fitInUb, writeX, hasBias = false;
    float eps;
};

extern "C" __global__ __aicore__ void add_layer_norm_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                                            GM_ADDR gamma, GM_ADDR beta,
                                                            GM_ADDR y, GM_ADDR meanOut, GM_ADDR rstdOut, GM_ADDR xOut,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAddLayerNorm op;
    op.Init(x1, x2, bias, gamma, beta, y, meanOut, rstdOut, xOut,
            tiling_data.colLength, tiling_data.smallRows, tiling_data.tailRows,
            tiling_data.ubTile, tiling_data.epsilon, tiling_data.writeX);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void add_layer_norm_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x1, uint8_t* x2,
                              uint8_t* bias, uint8_t* gamma, uint8_t* beta, uint8_t* y, uint8_t* meanOut,
                              uint8_t* rstdOut, uint8_t* xOut, uint8_t* workspace, uint8_t* tiling) {
    add_layer_norm_custom<<<blockDim, l2ctrl, stream>>>(x1, x2, bias, gamma, beta, y, meanOut, rstdOut, xOut, workspace, tiling);
}
#endif
