/*
 * Round 6 - host 侧：MultiCoreMatmulTiling 自动 tiling
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
// //?? matmul_tiling 头文件路径假设
#include "lib/matmul_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();

    auto xShape = context->GetInputShape(0)->GetStorageShape();
    auto wShape = context->GetInputShape(1)->GetStorageShape();
    uint32_t M = xShape.GetDim(0);
    uint32_t K = xShape.GetDim(1);
    uint32_t N = wShape.GetDim(1);

    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    tilingApi.SetDim(ascendcPlatform->GetCoreNumAic());
    tilingApi.SetAType(AscendC::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetBType(AscendC::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetCType(AscendC::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(AscendC::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetBufferSpace(-1, -1, -1);
    tilingApi.EnableBias(true);

    // //?? #1: TCubeTiling 序列化方式——假设 GetTiling 直接写入 RawTilingData 的 buffer
    matmul_tiling::TCubeTiling tilingData;
    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    // workspace = 系统 workspace（Matmul 内部实现需要）
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

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
class MatmulCustom : public OpDef {
public:
    explicit MatmulCustom(const char* name) : OpDef(name) {
        this->Input("a").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(MatmulCustom);
}  // namespace ops
