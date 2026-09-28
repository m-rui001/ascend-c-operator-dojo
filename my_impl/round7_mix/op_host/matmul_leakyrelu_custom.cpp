/*
 * Round 7 - host 侧：MIX 场景的 tiling（SetCType=LCM、SetDim/SetBlockDim 分离规则）
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "lib/matmul_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shapeA = context->GetInputShape(0)->GetOriginShape();
    auto shapeB = context->GetInputShape(1)->GetOriginShape();
    int32_t M = shapeA.GetDim(0);
    int32_t K = shapeA.GetDim(1);
    int32_t N = shapeB.GetDim(1);

    matmul_tiling::MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    // C 留在片上 UB（LCM==VECCALC），由 AIV 激活后再写 GM
    cubeTiling.SetCType(matmul_tiling::TPosition::LCM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    // 分离模式：SetDim = AIV 数；SetBlockDim = AI Core(AIC+AIV 组) 数
    // //?? 换算关系：blockDim = coreNumAic 组，每组 2 AIV，SetDim = blockDim * 2？待销案
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    cubeTiling.SetDim(aicNum * 2);

    matmul_tiling::TCubeTiling tilingData;
    if (cubeTiling.GetTiling(tilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    context->SetBlockDim(aicNum);

    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    auto a = context->GetInputShape(0);
    auto b = context->GetInputShape(1);
    auto c = context->GetOutputShape(0);
    c->SetDimNum(2);
    c->SetDim(0, a->GetDim(0));
    c->SetDim(1, b->GetDim(1));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MatmulLeakyreluCustom : public OpDef {
public:
    explicit MatmulLeakyreluCustom(const char* name) : OpDef(name) {
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(MatmulLeakyreluCustom);
}  // namespace ops
