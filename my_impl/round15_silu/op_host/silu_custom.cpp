/*
 * Round 15 - host 侧：沿用 R12 模板（32B 锚定大小核 + dtype 长度表 + 模式A blockDim）
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t coreNumTotal = platform.GetCoreNum();

    auto dtype = context->GetInputDesc(0)->GetDataType();
    uint64_t dtypeLen = (dtype == ge::DT_FLOAT) ? 4 : 2;
    uint64_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint64_t blockCnt = (totalLength * dtypeLen + 31) / 32;

    uint64_t ubTile = ubSize / 4 / 2 / 32 * 32 / dtypeLen;   // 4 份预算
    if (ubTile == 0) return ge::GRAPH_FAILED;
    uint64_t coreNum = (ubTile >= totalLength) ? 1
        : (coreNumTotal < blockCnt ? coreNumTotal : blockCnt);
    uint64_t everyCoreBlock = blockCnt / coreNum;
    uint64_t tailCoreNum = blockCnt % coreNum;
    uint64_t smallCoreNum = everyCoreBlock * 32 / dtypeLen;
    uint64_t bigCoreNum = tailCoreNum ? (everyCoreBlock + 1) * 32 / dtypeLen : smallCoreNum;

    auto tiling = context->GetTilingData<TilingDataSilu>();
    tiling->totalLength = totalLength;
    tiling->smallCoreNum = smallCoreNum;
    tiling->bigCoreNum = bigCoreNum;
    tiling->tailCoreNum = tailCoreNum;
    tiling->ubTile = ubTile;
    context->SetBlockDim(coreNum);
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class SiluCustom : public OpDef {
public:
    explicit SiluCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SiluCustom);
}  // namespace ops
