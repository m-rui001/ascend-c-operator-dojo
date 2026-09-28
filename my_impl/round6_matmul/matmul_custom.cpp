/*
 * Round 6 - 我自己写的 Matmul 算子（Cube 通路，Matmul 高阶 API）
 * C = A[M,K] * B[K,N] (+bias)，A/B fp16 ND，C fp32 ND
 * 对 API 细节没把握处用 //?? 标注
 */
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                    GM_ADDR workspace, GM_ADDR tiling);

// //?? #1: TCubeTiling 如何进入 tiling 数据——先按"原始 buffer 直接透传"假设
struct MatmulCustomTiling {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t hasBias;
    // TCubeTiling 的二进制内容直接跟在本结构后（host 序列化）
};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);  // //?? 非框架工程路径：文档说需要手动设系统 workspace
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }

    // //?? #4: 纯 Cube 模式跑在 AIC 上，入口是否需要 KERNEL_TASK_TYPE 声明？
    // 简单场景：AIC-only 应该是默认行为，暂不加宏（对比环节销案）

    // //?? #1 续：直接把 GM 上的 tiling 内容当作 TCubeTiling 使用（假设 host 只序列化了 TCubeTiling）
    // 正确做法可能是 GET_TILING_DATA 后取内嵌字段，暂按此假设
    using AType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
    using BType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
    using CType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BiasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    AscendC::Matmul<AType, BType, CType, BiasType> mm;
    // //?? TCubeTiling 从 tiling GM 读出：无把握，先按"REGIST_MATMUL_OBJ 接收指针"写
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, tiling);

    mm.SetTensorA(reinterpret_cast<__gm__ half*>(a));
    mm.SetTensorB(reinterpret_cast<__gm__ half*>(b));
    if (bias != nullptr) {
        mm.SetBias(reinterpret_cast<__gm__ float*>(bias));
    }

    // 简单场景用 IterateAll 一步到位；迭代控制场景应改 while(mm.Iterate()) + GetTensorC
    mm.IterateAll(reinterpret_cast<__gm__ float*>(c));
    mm.End();
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                      uint8_t* a, uint8_t* b, uint8_t* bias, uint8_t* c, uint8_t* workspace, uint8_t* tiling) {
    matmul_custom<<<blockDim, l2ctrl, stream>>>(a, b, bias, c, workspace, tiling);
}
#endif
