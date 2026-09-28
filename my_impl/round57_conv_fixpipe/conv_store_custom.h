/*
 * Round 57 - Conv 出口模式 mini：NC1HWC0 分形偏移计算 + 原子/顺序双路分派
 * （Fixpipe 本体是硬件指令路径，mini 实现偏移与分派逻辑）
 */
#include "kernel_operator.h"

constexpr uint32_t C0_SIZE = 16;       // NC1HWC0 的 C0（fp16 分形宽）
constexpr uint32_t BLOCK_CUBE = 16;

// Intf Config 模拟：格式是编译期成员
struct ConvConfig {
    enum class Fmt { NC1HWC0, FRACTALZ_C04, ND };
    static constexpr Fmt format = Fmt::NC1HWC0;
    using DstT = half;
    using L0cT = float;
};

struct ConvCtx {
    uint64_t curNL0Idx_;   // N 方向 L0 块索引
    uint64_t curML0Idx_;   // M 方向 L0 块索引
    uint32_t stepN;        // N 方向 base 块步数
    uint32_t mIter_;       // M 方向 base 块迭代数
    uint32_t baseN;        // base 块 N 宽
    uint32_t baseM;        // base 块 M 高
    uint32_t Cout;         // 输出通道
    uint32_t baseUseN_, baseUseM_;  // 实际使用的 base 尺寸（尾块）
};

// 分形偏移：NC1HWC0 下 N、M 两级 base 块的线性偏移（生产 dstOffset 同型）
__aicore__ inline uint64_t FractalDstOffset(const ConvCtx& ctx)
{
    return (uint64_t)(ctx.curNL0Idx_ % ctx.stepN) * ctx.baseN * ctx.Cout +
           (uint64_t)(ctx.curML0Idx_ % ctx.mIter_) * ctx.baseM * C0_SIZE;
}

// 出口模式：原子（跨 batch 累加）vs 顺序（确定性），mini 以标志分派 + 偏移演示
enum class StoreMode : uint8_t { ATOMIC, SEQUENTIAL };

class MyConvStore {
public:
    __aicore__ inline void Init(GM_ADDR l0cStaging, GM_ADDR out, const ConvCtx& ctx, StoreMode mode)
    {
        this->ctx = ctx;
        this->mode = mode;
        outGm.SetGlobalBuffer((__gm__ half*)out, (uint64_t)ctx.Cout * (ctx.mIter_ * ctx.baseM) * (ctx.stepN * ctx.baseN));
        stageGm.SetGlobalBuffer((__gm__ half*)l0cStaging, ctx.baseN * ctx.baseM * C0_SIZE);
    }

    // 一个 base 块的写出决策：偏移计算 + 模式分派
    __aicore__ inline void StoreOne(const AscendC::LocalTensor<half>& l0cBlock)
    {
        uint64_t dstOffset = FractalDstOffset(ctx);
        if (mode == StoreMode::ATOMIC) {
            // 生产：SetAtomicAdd<DstT>() 后 Fixpipe 直写——多 batch 贡献同一块时累加
            AscendC::SetAtomicAdd<half>();
            AscendC::DataCopyPad(outGm[dstOffset], l0cBlock,
                                 AscendC::DataCopyExtParams{1, (uint32_t)(ctx.baseUseM_ * C0_SIZE * sizeof(half)), 0, 0, 0});
            AscendC::SetAtomicNone();
        } else {
            // 顺序写：各 batch 写各自 workspace 段，post 阶段聚合（确定性）
            AscendC::DataCopyPad(stageGm[dstOffset % (ctx.baseN * ctx.baseM * C0_SIZE)], l0cBlock,
                                 AscendC::DataCopyExtParams{1, (uint32_t)(ctx.baseUseM_ * C0_SIZE * sizeof(half)), 0, 0, 0});
        }
        // 推进 L0 块索引
        ctx.curML0Idx_ = (ctx.curML0Idx_ + 1) % ctx.mIter_;
        if (ctx.curML0Idx_ == 0) ctx.curNL0Idx_++;
    }

private:
    ConvCtx ctx;
    StoreMode mode;
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<half> outGm, stageGm;
};
