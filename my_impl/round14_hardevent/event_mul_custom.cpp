/*
 * Round 14 - 事件驱动版 Mul：TBuf 双槽 + 显式 HardEvent 事件对（脱离 TQue 的裸流水）
 * 亲手管理 MTE2/V/MTE3 三管依赖 + 槽复用(MTE3_MTE2) + 标量配置(S_V)
 * 对没把握的语义用 //?? 标注
 */
#include "kernel_operator.h"

constexpr uint64_t BLOCK_SIZE = 32;
constexpr int32_t SLOTS = 2;  // 双槽 ping-pong

class KernelEventMul {
public:
    __aicore__ inline KernelEventMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                uint64_t totalLength,
                                uint64_t smallCoreNum, uint64_t bigCoreNum, uint64_t tailCoreNum,
                                uint64_t ubTile, float scale)
    {
        uint64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < tailCoreNum) {
            coreNum = bigCoreNum; coreOffset = blockIdx * bigCoreNum;
        } else {
            coreNum = smallCoreNum; coreOffset = tailCoreNum * bigCoreNum + (blockIdx - tailCoreNum) * smallCoreNum;
        }
        this->ubTile = ubTile;
        tileNum = (coreNum + ubTile - 1) / ubTile;

        xGm.SetGlobalBuffer((__gm__ half*)x + coreOffset, coreNum);
        wGm.SetGlobalBuffer((__gm__ half*)w + coreOffset, coreNum);  // w 与 x 同布局（广播行场景应按 R5 行常驻）
        yGm.SetGlobalBuffer((__gm__ half*)y + coreOffset, coreNum);

        // 双槽：x/y 各一个大 TBuf 切两槽（//?: 偏移切槽 vs 两个 TBuf，本轮选大 TBuf+偏移）
        pipe.InitBuffer(xSlots, SLOTS * ubTile * sizeof(half));
        pipe.InitBuffer(wSlots, SLOTS * ubTile * sizeof(half));
        pipe.InitBuffer(ySlots, SLOTS * ubTile * sizeof(half));

        // S_V 场景：scale 经标量写入后供 V 使用（演示用，实际 Muls 直接带标量更优）
        scale_ = scale;
    }

    __aicore__ inline void Process()
    {
        if (coreNum == 0) return;
        AscendC::LocalTensor<half> xAll = xSlots.Get<half>();
        AscendC::LocalTensor<half> wAll = wSlots.Get<half>();
        AscendC::LocalTensor<half> yAll = ySlots.Get<half>();

        for (uint64_t t = 0; t < tileNum; t++) {
            int s = t % SLOTS;
            uint32_t count = (t == tileNum - 1) ? (uint32_t)(coreNum - t * ubTile) : (uint32_t)ubTile;
            uint64_t off = t * ubTile;

            // 槽复用：等上上轮的 MTE3 搬出完成（MTE3_MTE2），才允许本轮 CopyIn 覆盖该槽
            if (t >= SLOTS) {
                WaitFlag<HardEvent::MTE3_MTE2>(evSlot[s]);   // //?: Wait 即可，SetFlag 在 CopyOut 后挂出
            }

            // CopyIn (MTE2)
            AscendC::DataCopy(xAll[s * ubTile], xGm[off], count);
            AscendC::DataCopy(wAll[s * ubTile], wGm[off], count);

            // MTE2 -> V：搬入完成才能算
            event_t evMV = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(evMV);
            WaitFlag<HardEvent::MTE2_V>(evMV);

            // Compute (V)
            AscendC::Mul(yAll[s * ubTile], xAll[s * ubTile], wAll[s * ubTile], count);
            AscendC::PipeBarrier<PIPE_V>();
            if (scale_ != 1.0f) {
                // S_V 演示：标量先写 UB，再让 V 用（生产中直接 Muls 更优）
                AscendC::LocalTensor<float> sv = statBuf.Get<float>();
                sv.SetValue(0, scale_);                       // 标量写（S 管产生）
                event_t evSV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(evSV);
                WaitFlag<HardEvent::S_V>(evSV);
                AscendC::Muls(yAll[s * ubTile], yAll[s * ubTile], sv.GetValue(0), count);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // V -> MTE3：算完才能搬出
            event_t evVM = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(evVM);
            WaitFlag<HardEvent::V_MTE3>(evVM);

            // CopyOut (MTE3)
            AscendC::DataCopy(yGm[off], yAll[s * ubTile], count);

            // 槽释放：MTE3 完成后标记（下一轮同槽 CopyIn 前 Wait）
            evSlot[s] = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(evSlot[s]);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xSlots, wSlots, ySlots, statBuf;
    AscendC::GlobalTensor<half> xGm, wGm, yGm;
    event_t evSlot[SLOTS];
    uint64_t coreNum, coreOffset, ubTile, tileNum, cols_w;
    float scale_;
};

extern "C" __global__ __aicore__ void event_mul_custom(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelEventMul op;
    op.Init(x, w, y, tiling_data.totalLength,
            tiling_data.smallCoreNum, tiling_data.bigCoreNum, tiling_data.tailCoreNum,
            tiling_data.ubTile, tiling_data.scale);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void event_mul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                         uint8_t* x, uint8_t* w, uint8_t* y, uint8_t* workspace, uint8_t* tiling) {
    event_mul_custom<<<blockDim, l2ctrl, stream>>>(x, w, y, workspace, tiling);
}
#endif
