/*
 * Round 29 - R16 的 add_scalar 迁移到 v2 工厂形态（mini 复述版）
 * v2 代际特征：Tiling 泛型化 / TPipe 内置 / Predicate 对象引用 / needTempBuf 旗标
 */
#include "kernel_operator.h"

// v2 风格：运算符是 Predicate 对象（成员函数，可 static_assert 校验签名）
struct AddsPredicateV2 {
    template <typename T>
    __aicore__ inline void Compute(AscendC::LocalTensor<T>& out, const AscendC::LocalTensor<T>& in,
                                   AscendC::LocalTensor<float>& /*f32work*/, float scalar, int64_t count) {
        AscendC::Adds(out, in, (T)scalar, count);
    }
};

template <typename T, typename Pred, int32_t bufferNum = 2, bool needCopyOut = true,
          bool needTempBuf = false, typename Tiling = ForeachCommonTilingData>
class MyForeachAddScalarV2 : public /* KernelForeachElewise<T, ...> 模拟 */ MyElewiseSim<T, Tiling> {
public:
    __aicore__ inline MyForeachAddScalarV2(Pred& p) : Base(p) {}

    __aicore__ inline void Process()
    {
        // v2：主循环在 Elewise 基类；本类只需实现 Compute 钩子
        Base::Process();
    }

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount,
                                   AscendC::LocalTensor<float>& f32work, bool isRemainder)
    {
        AscendC::LocalTensor<T> in = Base::dataQueue.template DeQue<T>();
        AscendC::LocalTensor<T> out = Base::outQueue.template AllocTensor<T>();
        pred.Compute(out, in, f32work, scalarVal, dataCount);
        Base::outQueue.template EnQue(out);
        Base::dataQueue.template FreeTensor(in);
    }
    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart)
    {
        // //?: scalar 是单一标量——v2 无逐张量标量钩子需求时留空
    }
    float scalarVal = 0.0f;
    Pred& pred;
    using Base = MyElewiseSim<T, Tiling>;
};

// 标量经 tiling 下发时的入口（v2 风格）
extern "C" __global__ __aicore__ void foreach_add_scalar_v2_custom(GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
                                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachCommonTilingData);
    GET_TILING_DATA(tilingData, tiling);
    AddsPredicateV2 pred;
    if (TILING_KEY_IS(1)) {
        MyForeachAddScalarV2<half, AddsPredicateV2> op(pred);
        op.InitFromTiling(x, y, workspace, &tilingData, /*scalarFromTiling*/ tilingData.scalarPad);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        MyForeachAddScalarV2<float, AddsPredicateV2> op(pred);
        op.InitFromTiling(x, y, workspace, &tilingData, tilingData.scalarPad);
        op.Process();
    }
}
