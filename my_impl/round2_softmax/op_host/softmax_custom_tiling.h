#ifndef SOFTMAX_CUSTOM_TILING_H
#define SOFTMAX_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxTilingData)
TILING_DATA_FIELD_DEF(uint64_t, rowCount);     // 总行数
TILING_DATA_FIELD_DEF(uint64_t, colLength);    // 每行列数
TILING_DATA_FIELD_DEF(uint64_t, ubTile);       // 每片列数
TILING_DATA_FIELD_DEF(uint64_t, colTail);      // 尾片列数
TILING_DATA_FIELD_DEF(uint64_t, smallRows);    // 小核行数
TILING_DATA_FIELD_DEF(uint64_t, tailRows);     // 前 tailRows 个核每核多一行
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxTilingData)
}  // namespace optiling
#endif
