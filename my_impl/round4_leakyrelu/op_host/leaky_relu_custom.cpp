/*
 * Round 4 - host 侧 tiling：elementwise 一维切分
 * 32B 块均分 + 大小核 + UB 反推 ubTile + attr(negativeSlope) 下发
 */
#include "leaky_relu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LeakyReluTilingData tiling;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t coreNumTotal = platform.GetCoreNum();

    auto dtype = context->GetInputDesc(0)->GetDataType();
    uint64_t dtypeLen = (dtype == ge::DT_FLOAT) ? 4 : 2;

    uint64_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint64_t totalBytes = totalLength * dtypeLen;
    uint64_t totalBytesAlign = (totalBytes + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    // UB 预算：x 队列(2份×double) + z 队列(2份×double) = 4 份
    uint64_t ubTileBytes = ubSize / 4 / BUFFER_NUM / BLOCK_SIZE * BLOCK_SIZE;
    uint64_t ubTile = ubTileBytes / dtypeLen;
    if (ubTile == 0) return ge::GRAPH_FAILED;

    uint64_t coreNum = coreNumTotal;
    if (ubTile >= totalLength) {
        coreNum = 1;
    } else {
        // 每核至少 32B：核数上限 = 32B 块数
        uint64_t blockCnt = totalBytesAlign / BLOCK_SIZE;
        coreNum = coreNumTotal < blockCnt ? coreNumTotal : blockCnt;
    }

    // 大小核：32B 块均分，余数给前 tailCoreNum 个核
    uint64_t everyCoreBlock = blockCnt / coreNum;
    uint64_t tailCoreNum = blockCnt % coreNum;
    uint64_t smallCoreNum = everyCoreBlock * BLOCK_SIZE / dtypeLen;
    uint64_t bigCoreNum = smallCoreNum;
    if (tailCoreNum != 0) {
        bigCoreNum = (everyCoreBlock + 1) * BLOCK_SIZE / dtypeLen;
    }

    tiling.set_totalLength(totalLength);
    tiling.set_smallCoreNum(smallCoreNum);
    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_tailCoreNum(tailCoreNum);
    tiling.set_ubTile(ubTile);
    // attr 下发
    auto attrs = context->GetAttrs();
    float slope = 0.0f;
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        slope = attrs->GetFloat(0);
    }
    tiling.set_negativeSlope(slope);
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
class LeakyReluCustom : public OpDef {
public:
    explicit LeakyReluCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("negative_slope")
            .AttrType(OPTIONAL)
            .Float(0.0f);  // 默认 0 即 ReLU
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b").AddConfig("ascend310b");
    }
};
OP_ADD(LeakyReluCustom);
}  // namespace ops
