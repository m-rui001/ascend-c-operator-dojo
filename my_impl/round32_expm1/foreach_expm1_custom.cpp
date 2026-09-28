/*
 * Round 32 - 我实现的 foreach_expm1：基础 API 组合（Exp + Adds(-1)）
 */
#include "kernel_operator.h"
#include "foreach/foreach_one_scalar_binary.h"

using namespace AscendC;
using namespace Common::OpKernel;

template <typename T>
__aicore__ void MyExpm1Adapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& uValue) {
    PipeBarrier<PIPE_V>();
    Exp(dstLocal, srcLocal, uValue);            // e^x
    PipeBarrier<PIPE_V>();
    Adds(dstLocal, srcLocal, T(-1), uValue);    // 原地减 1（隐式输出同缓冲）
}

extern "C" __global__ __aicore__ void foreach_expm1_custom(GM_ADDR x, GM_ADDR y,
                                                           GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachImplictOutput<half, half, MyExpm1Adapter<half>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachImplictOutput<float, float, MyExpm1Adapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachImplictOutput<bfloat16_t, float, MyExpm1Adapter<float>, 2, 1> op;  // bf16 升 fp32 域
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
#endif
}
