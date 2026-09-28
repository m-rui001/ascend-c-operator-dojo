/*
 * Round 31 - 我实现的 foreach_log：工厂 + 一行 Adapter（超越函数的极简形态）
 * 验证：有基础 API 指令封装的超越函数，工厂实例化即完整算子
 */
#include "kernel_operator.h"
#include "foreach/foreach_one_scalar_binary.h"  // //?: include 路径按工程布局，实际用 foreach_implict_output.h

using namespace AscendC;
using namespace Common::OpKernel;

template <typename T>
__aicore__ void MyLogAdapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& uValue) {
    Log<T>(dstLocal, srcLocal);   // 基础 API 指令封装（对数硬件指令）
}

extern "C" __global__ __aicore__ void foreach_log_custom(GM_ADDR x, GM_ADDR y,
                                                         GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachImplictOutput<half, half, MyLogAdapter<half>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachImplictOutput<float, float, MyLogAdapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        // bf16 进、fp32 域算、基类回转 bf16（进出 dtype 分离的模板签名）
        ForeachImplictOutput<bfloat16_t, float, MyLogAdapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
}
