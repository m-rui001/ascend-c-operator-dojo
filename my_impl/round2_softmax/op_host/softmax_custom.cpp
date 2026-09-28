/*
 * Round 2 - host 侧 tiling（按 op_host 工程结构，Round 1 教训）
 * 行为单位做大小核切分；ubTile 由 UB 容量反推
 */
#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SoftmaxTilingData tiling;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t coreNum = platform.GetCoreNum();

    uint64_t rowCount = context->GetInputShape(0)->GetStorageShape().GetShapeSize() /
                        context->GetInputShape(0)->GetStorageShape().GetDim(context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1);
    uint64_t colLength = context->GetInputShape(0)->GetStorageShape().GetDim(
        context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1);

    uint32_t dataTypeLength = 2;  // fp16 简化；生产应查询 dtype 长度
    // UB 预算：x 队列2 + y 队列2 + tmp 2份 = 6 份 double buffer
    uint64_t perTileBytes = ubSize / 6;
    uint64_t ubTile = (perTileBytes / dataTypeLength / BLOCK_SIZE) * BLOCK_SIZE / dataTypeLength;
    if (ubTile == 0) return ge::GRAPH_FAILED;
    if (ubTile > colLength) ubTile = colLength;

    uint64_t colTail = colLength % ubTile == 0 ? ubTile : colLength % ubTile;

    // 行为单位的大小核切分
    uint64_t smallRows = rowCount / coreNum;
    uint64_t tailRows = rowCount % coreNum;
    if (smallRows == 0) { coreNum = rowCount; }  // 行数少于核数：一核一行，多余核空跑

    tiling.set_rowCount(rowCount);
    tiling.set_colLength(colLength);
    tiling.set_ubTile(ubTile);
    tiling.set_colTail(colTail);
    tiling.set_smallRows(smallRows);
    tiling.set_tailRows(tailRows);
    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SoftmaxCustom);
}  // namespace ops
