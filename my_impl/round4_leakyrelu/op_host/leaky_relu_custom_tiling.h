#ifndef LEAKY_RELU_CUSTOM_TILING_H
#define LEAKY_RELU_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LeakyReluTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint64_t, smallCoreNum);   // 小核元素数
TILING_DATA_FIELD_DEF(uint64_t, bigCoreNum);     // 大核元素数
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);    // 大核个数
TILING_DATA_FIELD_DEF(uint64_t, ubTile);         // 每片元素数
TILING_DATA_FIELD_DEF(float, negativeSlope);     // 标量属性
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(LeakyReluCustom, LeakyReluTilingData)
}  // namespace optiling
#endif
