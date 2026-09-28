/*
 * Round 52 - 我实现的 FA Grad mini：消费正向 softmax 状态 → dS → 三路 matmul（抽象）
 */
#include "kernel_operator.h"

constexpr int32_t BLOCK_ALIGN = 32;

class MyFAGrad {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR dy,
                                GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR p,
                                GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                uint32_t s1, uint32_t s2, uint32_t d, GM_ADDR tiling)
    {
        this->s1 = s1; this->s2 = s2; this->d = d;
        this->dAlign = (d * sizeof(float) + BLOCK_ALIGN - 1) / BLOCK_ALIGN * 8;
        blockIdx = AscendC::GetBlockIdx();
        qGm.SetGlobalBuffer((__gm__ float*)q, (uint64_t)s1 * d);
        kGm.SetGlobalBuffer((__gm__ float*)k, (uint64_t)s2 * d);
        vGm.SetGlobalBuffer((__gm__ float*)v, (uint64_t)s2 * d);
        dyGm.SetGlobalBuffer((__gm__ float*)dy, (uint64_t)s1 * d);
        pGm.SetGlobalBuffer((__gm__ float*)p, (uint64_t)s1 * s2);       // 正向 P（或由 S+state 重算）
        maxGm.SetGlobalBuffer((__gm__ float*)softmaxMax, s1 * 8);        // [S1,8] 布局契约
        sumGm.SetGlobalBuffer((__gm__ float*)softmaxSum, s1 * 8);
        dqGm.SetGlobalBuffer((__gm__ float*)dq, (uint64_t)s1 * d);
        dkGm.SetGlobalBuffer((__gm__ float*)dk, (uint64_t)s2 * d);
        dvGm.SetGlobalBuffer((__gm__ float*)dv, (uint64_t)s2 * d);

        pipe.InitBuffer(rowBuf, 1, dAlign * sizeof(float));
        pipe.InitBuffer(dsBuf, 1, s2 > BLOCK_ALIGN ? ((s2 + 63) / 64) * 64 * sizeof(float) : 64 * sizeof(float));
        pipe.InitBuffer(stateBuf, 1, 16 * sizeof(float));                // m/sum 各 8
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> state = stateBuf.Get<float>();
        AscendC::LocalTensor<float> dsRow = dsBuf.Get<float>();
        // [S1,8] 布局的 softmax 状态读回（R51 契约的消费侧）
        AscendC::DataCopyParams s8{1, (uint16_t)(8 * sizeof(float)), 0, 0};
        AscendC::DataCopyPadExtParams<float> spad{false, 0, 0, 0};

        for (uint32_t r = 0; r < s1; r++) {
            // ---- 读回该行状态（8-float 步长布局）----
            AscendC::DataCopyPad(state, sumGm[r * 8], s8, spad);
            AscendC::DataCopyPad(state[8], maxGm[r * 8], s8, spad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // softmaxSum 与 P 已含缩放（正向出口）；dS = softmaxGrad(P, dY) —— 生产用
            // SoftmaxGradFront 高阶指令一条完成（//?: mini 以逐元素示意）
            AscendC::LocalTensor<float> pRow = dsBuf.Get<float>()[0];  // P 行（示意：驻留段省略）
            AscendC::LocalTensor<float> dyRow = rowBuf.Get<float>();
            AscendC::DataCopy(dyRow, dyGm[(uint64_t)r * d], dAlign);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            // dS 行 = P ⊙ (dO·V^T − Σ(dO⊙O))——示意级逐块（生产 SoftmaxGradFront）
            // dqRow = dS · K块（生产：Matmul bTypeTranspose / MM34 变体）
            // 省略内部三 matmul 细节（与 R51 同级抽象）

            // dV = P^T · dO、dK = dS^T · Q：转置 matmul（bTypeTranspose）
            // dQ 部分和 → workspace → SyncAll×2 聚合（确定性，生产 3074/3111 两次 SyncAll）
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowBuf, dsBuf, stateBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, dyGm, pGm, maxGm, sumGm, dqGm, dkGm, dvGm;
    uint32_t s1, s2, d, dAlign;
    uint16_t blockIdx;
};
