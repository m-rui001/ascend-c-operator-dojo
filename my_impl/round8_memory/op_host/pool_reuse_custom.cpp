/*
 * Round 8 - host 侧：单核演示（本专题重点是 UB 内存语义，多核切分沿用前几轮模式）
 */
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // //?? 演示工程：直接用 TILING_DATA 结构（字段 totalLength/stageLength）
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(1);  // 单核：专题聚焦 UB 语义
    // tiling 结构见 tiling 头文件
    auto tiling = context->GetTilingData<TilingDataPoolReuse>();
    tiling->totalLength = totalLength;
    tiling->stageLength = totalLength / 2;
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class PoolReuseCustom : public OpDef {
public:
    explicit PoolReuseCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(PoolReuseCustom);
}  // namespace ops
