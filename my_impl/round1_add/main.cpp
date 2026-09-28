/*
 * Round 1 - Host 侧调用程序（我自己的写法）
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"

struct AddTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddTilingData tiling);

std::vector<float> kernel_add(std::vector<float>& x, std::vector<float>& y) {
    constexpr uint32_t numBlocks = 8;
    uint32_t totalLength = x.size();
    size_t totalByteSize = totalLength * sizeof(float);
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AddTilingData tiling = {totalLength, 8};

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    uint8_t *xDevice = nullptr, *yDevice = nullptr, *zDevice = nullptr;
    uint8_t* zHost = nullptr;
    aclrtMallocHost((void**)(&zHost), totalByteSize);
    aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    add_custom<<<numBlocks, nullptr, stream>>>(xDevice, yDevice, zDevice, tiling);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<float> z((float*)zHost, (float*)(zHost + totalByteSize));

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFreeHost(zHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return z;
}

int main() {
    constexpr uint32_t totalLength = 8 * 2048;
    std::vector<float> x(totalLength, 1.2f);
    std::vector<float> y(totalLength, 2.3f);
    std::vector<float> z = kernel_add(x, y);

    bool ok = true;
    for (uint32_t i = 0; i < totalLength; i++) {
        if (z[i] != x[i] + y[i]) {
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "[Success] verify passed" : "[Failed] verify failed") << std::endl;
    return ok ? 0 : 1;
}
