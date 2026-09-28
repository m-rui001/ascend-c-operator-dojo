/*
 * Round 54 - FA 变体多核切分对比 mini：线性任务号 → 5 维轴索引的可配置分解
 * 轴序即切分策略：s1s2_bn2gs1 = [B,N2,G,S1,S2](S1 外) vs bn2gs1s2_b = [B,N2,G,S1,S2](B 外直接映射)
 */
#include "kernel_operator.h"

struct AxisOrder {
    uint32_t B, N2, G, S1, S2;   // 各轴的任务数（base 块数）
};

// 轴序枚举：线性任务号分解时轴的优先序（外层在前）
enum class AxisPriority : uint8_t {
    BN2_G_S1_S2,   // bn2gs1s2_b：B,N2 外层 → boIdx 直映射
    S1S2_BN2_G,    // s1s2_bn2gs1：S1,S2 内层展开（sparse 可跳块）
};

struct AxisIdx {
    uint32_t b, n2, g, s1, s2;
};

// 线性任务号 → 5 维轴索引（div/mod 链，R17 CalcOffset 的 5 轴泛化）
__aicore__ inline AxisIdx Decompose(uint64_t taskId, const AxisOrder& order, AxisPriority prio)
{
    AxisIdx a{};
    if (prio == AxisPriority::BN2_G_S1_S2) {
        // 外层 B：boIdx = taskId 直映射（生产 B 变体的 boIdx 直接取 taskId）
        uint64_t rem = taskId;
        a.b = rem / (order.N2 * order.G * order.S1 * order.S2);
        rem %= order.N2 * order.G * order.S1 * order.S2;
        a.n2 = rem / (order.G * order.S1 * order.S2);
        rem %= order.G * order.S1 * order.S2;
        a.g = rem / (order.S1 * order.S2);
        rem %= order.S1 * order.S2;
        a.s1 = rem / order.S2;
        a.s2 = rem % order.S2;
    } else {
        // 内层 S1×S2：s1Outer 在前，sparse 场景按 GetS1LoopRange 裁剪
        uint64_t rem = taskId;
        a.s1 = rem / (order.S2 * order.G * order.N2 * order.B);
        rem %= order.S2 * order.G * order.N2 * order.B;
        a.s2 = rem / (order.G * order.N2 * order.B);
        rem %= order.G * order.N2 * order.B;
        a.g = rem / (order.N2 * order.B);
        rem %= order.N2 * order.B;
        a.n2 = rem / order.B;
        a.b = rem % order.B;
    }
    return a;
}

// 尾部空任务：任务数 = 有效任务 + 流水深度（extraInfo[3] → +2 或 +3）
__aicore__ inline uint64_t TotalTasksWithDrain(uint64_t validTasks, uint32_t pipelineDepth)
{
    return validTasks + (pipelineDepth - 1);
}

__aicore__ inline void MySplitDemo()
{
    // 两种轴序下任务 5 的分解对比（B=2,N2=2,G=1,S1=4,S2=4）
    AxisOrder order{2, 2, 1, 4, 4};
    AxisIdx a1 = Decompose(5, order, AxisPriority::BN2_G_S1_S2);
    AxisIdx a2 = Decompose(5, order, AxisPriority::S1S2_BN2_G);
    // a1: B 外层 → b=0,n2=1,g=0,s1=1,s2=1
    // a2: S1 外层 → s1=0,s2=1,g=1,n2=1,b=0
    // 相邻任务的数据局部性完全不同：a1 相邻共享 B/N2（KV 局部性）；a2 相邻共享 S1 行（Q 局部性）
}
