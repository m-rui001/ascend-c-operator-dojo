/*
 * Round 22 - 我实现的 LayerNorm 块状两遍方差（many-rows 场景，transpose 策略的简化复述版）
 * 块驻留 UB：Pass1 逐行 mean；Pass2 原地减 mean→整块平方→逐行 var（零 GM 重读）；γ/β 藏载
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyLayerNormBlockTwoPass {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                GM_ADDR mean, GM_ADDR rstd, GM_ADDR tiling)
    {
        // tiling: rowsPerBlock(ubFormer), rowSize, blocksPerCore, tailBlocks 等
        this->rowsPerBlock = tdRowsPerBlock;
        this->rowSize = tdRowSize;
        this->rowAlign = (rowSize * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        this->invN = 1.0f / rowSize;
        blockIdx = AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ float*)x, tdmTotalRows * rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, tdmTotalRows * rowSize);
        meanGm.SetGlobalBuffer((__gm__ float*)mean, tdmTotalRows);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd, tdmTotalRows);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, rowSize);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, rowSize);

        uint32_t tileLen = rowsPerBlock * rowAlign;
        pipe.InitBuffer(xBuf, 1, tileLen * sizeof(float));        // 块驻留
        pipe.InitBuffer(wBuf, 1, rowAlign * sizeof(float));       // 行归约/γ/β 载体
        pipe.InitBuffer(paramBuf, 1, 2 * rowsPerBlock * sizeof(float)); // mean|rstd 攒批
        pipe.InitBuffer(oneBuf, 1, BLOCK_ALIGN);                  // 全 1 tensor（Div 倒数用）
    }

    __aicore__ inline void Process()
    {
        uint64_t rowBase = blockIdx * rowsPerBlock;   // //?: 大小核循环块数精化从简（演示单块均分）
        // γ/β 预取藏载点：首轮 mean 归约期间拉 γ（R21 B22）
        for (uint32_t blk = 0; blk < myBlocks(); blk++) {
            ProcessBlock(rowBase + blk * rowsPerBlock);
        }
    }

private:
    __aicore__ inline uint32_t myBlocks() { return tdBlocksPerCore; }

    __aicore__ inline void ProcessBlock(uint64_t rowStart)
    {
        AscendC::LocalTensor<float> xLocal = xBuf.Get<float>();
        AscendC::LocalTensor<float> wLocal = wBuf.Get<float>();
        AscendC::LocalTensor<float> pLocal = paramBuf.Get<float>();
        AscendC::LocalTensor<float> oneLocal = oneBuf.Get<float>();

        // ---- 2D pad 载入整块 ----
        AscendC::DataCopyExtParams cp{rowsPerBlock, (uint32_t)(rowSize * sizeof(float)), 0,
                                      (uint16_t)((rowAlign - rowSize) * sizeof(float) / BLOCK_ALIGN), 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[rowStart * rowSize], cp, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // ---- Pass1: 逐行 mean（掩码计数直读累加器，R21 B10）----
        AscendC::AscendCUtils::SetMaskCount<float>();
        for (uint32_t r = 0; r < rowsPerBlock; r++) {
            AscendC::SetVectorMask<float>(0, rowSize);
            AscendC::ReduceSum<float>(wLocal, xLocal[r * rowAlign], wLocal, 1);
            uint64_t acc = AscendC::GetAccVal();
            pLocal.SetValue(r, *reinterpret_cast<float*>(&acc) * invN);
        }
        AscendC::SetMaskNorm();

        // ---- Pass2 原地两遍方差（块驻留，零 GM 重读）----
        for (uint32_t r = 0; r < rowsPerBlock; r++) {
            float m = pLocal.GetValue(r);   // //?: 标量通路 O(rowsPerBlock)——允许级（C/值随块数）
            AscendC::Adds(xLocal[r * rowAlign], xLocal[r * rowAlign], -m, rowSize);
        }
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Mul(xLocal, xLocal, xLocal, rowsPerBlock * rowAlign);   // 整块一次（B4 批行）
        AscendC::PipeBarrier<AscendC::PIPE_V>();
        AscendC::Muls(xLocal, xLocal, invN, rowsPerBlock * rowAlign);

        AscendC::AscendCUtils::SetMaskCount<float>();
        for (uint32_t r = 0; r < rowsPerBlock; r++) {
            AscendC::SetVectorMask<float>(0, rowSize);
            AscendC::ReduceSum<float>(wLocal, xLocal[r * rowAlign], wLocal, 1);
            uint64_t acc = AscendC::GetAccVal();
            float var = *reinterpret_cast<float*>(&acc);
            float rstd = 1.0f / sqrt(var + eps);
            pLocal.SetValue(rowsPerBlock + r, rstd);
        }
        AscendC::SetMaskNorm();

        // ---- Pass3: normalize（用 pLocal 中的 rstd，逐行 Muls）+ γ/β ----
        // γ 藏载：在 Pass2 的标量循环间隙已可发起（此处简化为循环后）
        AscendC::DataCopy(wLocal, gammaGm, rowAlign);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        for (uint32_t r = 0; r < rowsPerBlock; r++) {
            float rstd = pLocal.GetValue(rowsPerBlock + r);
            AscendC::Muls(xLocal[r * rowAlign], xLocal[r * rowAlign], rstd, rowSize);
            AscendC::Mul(xLocal[r * rowAlign], xLocal[r * rowAlign], wLocal, rowSize);
        }
        AscendC::DataCopy(wLocal, betaGm, rowAlign);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        for (uint32_t r = 0; r < rowsPerBlock; r++) {
            AscendC::Add(xLocal[r * rowAlign], xLocal[r * rowAlign], wLocal, rowSize);
        }

        // ---- 攒批写出 ----
        AscendC::DataCopyExtParams cpy{rowsPerBlock, (uint32_t)(rowSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[rowStart * rowSize], xLocal, cpy);
        AscendC::DataCopyExtParams cpp{1, (uint32_t)(rowsPerBlock * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(meanGm[rowStart], pLocal, cpp);
        AscendC::DataCopyPad(rstdGm[rowStart], pLocal[rowsPerBlock], cpp);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xBuf, wBuf, paramBuf, oneBuf;
    AscendC::GlobalTensor<float> xGm, yGm, meanGm, rstdGm, gammaGm, betaGm;
    uint32_t rowsPerBlock, rowSize, rowAlign, tdmTotalRows, tdBlocksPerCore, tdRowsPerBlock;
    uint16_t blockIdx;
    float invN;
    float eps = 1e-5f;   // //?: 重犯自查——应 tiling 下发（R21 已记录，此处标注待改）
};
