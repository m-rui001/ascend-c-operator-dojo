/*
 * Round 7 - 我自己写的 Matmul+LeakyRelu MIX 融合算子（范式路径，框架管 AIC/AIV 同步）
 * c = LeakyRelu(a*b + bias, alpha)
 * 依据：官方《融合算子-基础知识/算子实现》；对没把握处用 //?? 标注
 */
#include "lib/matmul_intf.h"
#include "kernel_operator.h"

constexpr float ALPHA = 0.01f;
constexpr int32_t DEFAULT_C0_SIZE = 16;  // //?? C0=32B/2B(half)=16? fp32 C 下应为 32B/4B=8，待销案

template <typename aType, typename bType, typename cType, typename biasType>
class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                AscendC::tiling::TCubeTiling tiling, AscendC::TPipe* pipe)
    {
        this->tiling = tiling;
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType*>(a));
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType*>(b));
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType*>(bias));
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType*>(c));
        this->pipe = pipe;
        pipe->InitBuffer(reluOutQueue, 2, tiling.baseM * tiling.baseN * sizeof(cType));  // //?? 双缓冲份数
    }

    __aicore__ inline void Process(AscendC::TPipe* pipe)
    {
        matmulObj.SetTensorA(aGlobal);
        matmulObj.SetTensorB(bGlobal);
        matmulObj.SetBias(biasGlobal);
        uint32_t computeRound = 0;
        while (matmulObj.template Iterate<true>()) {  // //?? <true> 语义待销案
            MatmulCompute();
            LeakyReluCompute();
            CopyOut(computeRound);
            computeRound++;
        }
        matmulObj.End();
    }

private:
    __aicore__ inline void MatmulCompute()
    {
        reluOutLocal = reluOutQueue.AllocTensor<cType>();
        // Cube 结果搬上 Vector 核（CO2->VECIN）
        matmulObj.template GetTensorC<true>(reluOutLocal, false, true);  // //?? 实参语义待销案
    }

    __aicore__ inline void LeakyReluCompute()
    {
        // //?? 尾块：最后一轮 Iterate 时 baseM*baseN 可能大于实际剩余——多算是否越界，待销案
        AscendC::LeakyRelu<cType>(reluOutLocal, reluOutLocal, (cType)ALPHA, tiling.baseM * tiling.baseN);
        reluOutQueue.EnQue(reluOutLocal);
    }

    __aicore__ inline void CopyOut(uint32_t count)
    {
        reluOutQueue.DeQue<cType>();
        uint32_t roundM = tiling.singleCoreM / tiling.baseM;
        uint32_t roundN = tiling.singleCoreN / tiling.baseN;
        // //?? iterateOrder=0 假设下 count→(m,n) 偏移映射
        uint32_t startOffset = (count % roundM) * tiling.baseM * tiling.N + (count / roundM) * tiling.baseN;
        AscendC::DataCopyParams copyParam = {
            (uint16_t)tiling.baseM,
            (uint16_t)(tiling.baseN * sizeof(cType) / DEFAULT_C0_SIZE),
            0,
            (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / DEFAULT_C0_SIZE)
        };
        AscendC::DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
        reluOutQueue.FreeTensor(reluOutLocal);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::GlobalTensor<aType> aGlobal, bGlobal, cGlobal, biasGlobal;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> reluOutQueue;  // //?? position 用 VECOUT 还是 VECIN？
    AscendC::LocalTensor<cType> reluOutLocal;
    Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
           AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           AscendC::MatmulType<AscendC::TPosition::LCM, CubeFormat::ND, cType>,
           AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>> matmulObj;
    AscendC::tiling::TCubeTiling tiling;
};

// mix 场景：AIC:AIV = 1:2；KFC workspace 由框架管理
__global__ __mix__(1, 2) void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                      __kfc_workspace__ GM_ADDR workspace,
                                                      AscendC::tiling::TCubeTiling tiling)
{
    AscendC::TPipe pipe;
    MatmulLeakyKernel<half, half, float, float> op;
    op.Init(a, b, bias, c, tiling, &pipe);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &op.tiling);  // //?? 成员私有：示意写法
    op.Process(&pipe);
}
