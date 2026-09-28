/*
 * Round 12 - host 侧：32B 锚定大小核切分 + approximate attr→TilingKey
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t coreNumTotal = platform.GetCoreNum();

    auto dtype = context->GetInputDesc(0)->GetDataType();
    uint64_t dtypeLen = (dtype == ge::DT_FLOAT) ? 4 : 2;   // CHECKLIST C1：dtype 长度表
    uint64_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint64_t totalBytes = totalLength * dtypeLen;
    uint64_t blockCnt = (totalBytes + BLOCK_SIZE - 1) / BLOCK_SIZE;   // 32B 锚点（CHECKLIST C2）

    uint64_t ubTile = ubSize / 4 / BUFFER_NUM / BLOCK_SIZE * BLOCK_SIZE / dtypeLen;  // 4 份预算（A7）
    if (ubTile == 0) return ge::GRAPH_FAILED;

    uint64_t coreNum = coreNumTotal;
    if (ubTile >= totalLength) {
        coreNum = 1;
    } else {
        coreNum = coreNumTotal < blockCnt ? coreNumTotal : blockCnt;  // 每核至少 32B
    }
    uint64_t everyCoreBlock = blockCnt / coreNum;
    uint64_t tailCoreNum = blockCnt % coreNum;
    uint64_t smallCoreNum = everyCoreBlock * BLOCK_SIZE / dtypeLen;
    uint64_t bigCoreNum = tailCoreNum ? (everyCoreBlock + 1) * BLOCK_SIZE / dtypeLen : smallCoreNum;

    auto tiling = context->GetTilingData<TilingDataGelu>();
    tiling->totalLength = totalLength;
    tiling->smallCoreNum = smallCoreNum;
    tiling->bigCoreNum = bigCoreNum;
    tiling->tailCoreNum = tailCoreNum;
    tiling->ubTile = ubTile;
    context->SetBlockDim(coreNum);

    // approximate attr → TilingKey 路由（CHECKLIST C3/C4）
    auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        auto mode = attrs->GetStr(0);
        context->SetTilingKey((mode != nullptr && std::string(mode) == "tanh") ? 2 : 1);
    } else {
        context->SetTilingKey(1);
    }

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class GeluCustom : public OpDef {
public:
    explicit GeluCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("approximate").AttrType(OPTIONAL).String("none");
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(GeluCustom);
}  // namespace ops
