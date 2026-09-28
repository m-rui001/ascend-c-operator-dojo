/*
 * Round 11 - host 侧：行数切分 + UB 反推 + epsilon 属性
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t coreNum = platform.GetCoreNum();

    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint64_t dims = shape.GetDimNum();
    uint64_t cols = shape.GetDim(dims - 1);
    uint64_t rows = shape.GetShapeSize() / cols;

    // UB 预算：x2份×双缓冲 + y2份×双缓冲 + fp32 work 2 + gamma 1 + rstd 小片 ≈ 按 10 份估
    uint64_t ubTile = ubSize / 10 / sizeof(half) / BLOCK_SIZE * BLOCK_SIZE;
    if (ubTile == 0) return ge::GRAPH_FAILED;
    if (ubTile > cols) ubTile = cols;

    uint64_t smallRows = rows / coreNum;
    uint64_t tailRows = rows % coreNum;
    if (smallRows == 0) { coreNum = rows; smallRows = 1; tailRows = 0; }

    auto tiling = context->GetTilingData<TilingDataRmsNorm>();
    tiling->colLength = cols;
    tiling->smallRows = smallRows;
    tiling->tailRows = tailRows;
    tiling->ubTile = ubTile;
    const float* eps = context->GetAttrs()->GetFloat(0);
    tiling->epsilon = *eps;
    context->SetBlockDim(coreNum);   // 检查项3：模式A
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;                        // 检查项4
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class RmsNormCustom : public OpDef {
public:
    explicit RmsNormCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("rstd").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(RmsNormCustom);
}  // namespace ops
