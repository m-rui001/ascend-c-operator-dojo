/*
 * Round 16 - host 侧：foreach_add_scalar tiling
 * 关键假设：host 可经 GetInputPtr 取得各张量设备地址，将描述表写入 workspace
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = platform.GetCoreNumAiv();

    // //?: 张量列表的遍历接口——动态输入句柄是猜的
    uint32_t tensorNum = context->GetDynamicInputShape(0, 0) != nullptr ? 0 : 0;  // 占位：真实接口待销案
    // 假设可枚举：for (i...) shape = context->GetDynamicInputShape(0, i)

    uint32_t dtypeLen = 2;
    uint64_t ubTile = (ubSize / 4 / BUFFER_NUM / BLOCK_SIZE) * BLOCK_SIZE / dtypeLen;

    // 描述表放 user workspace（地址 host 写不进设备—— //?: 真实机制待销案，可能走 tiling 数组或框架填表）
    size_t descBytes = tensorNum * 16;  // {addr, count} 各 8B
    size_t sysWs = platform.GetLibApiWorkSpaceSize();
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = descBytes + sysWs;

    auto tiling = context->GetTilingData<ForeachTilingData>();
    tiling->tensorNum = tensorNum;
    tiling->ubTile = ubTile;
    tiling->descTableAddr = 0;  // //?: workspace 基址在 tiling 阶段拿不到，机制待销案
    tiling->dtypeKey = 1;

    context->SetBlockDim(coreNum);
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class ForeachAddScalarCustom : public OpDef {
public:
    explicit ForeachAddScalarCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("scalar").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(ForeachAddScalarCustom);
}  // namespace ops
