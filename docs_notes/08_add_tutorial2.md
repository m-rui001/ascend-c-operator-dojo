---
title: Add自定义算子开发
description: "本入门教程，将会引导你完成以下任务，体验Ascend C SIMT算子开发基本流程。"
url: https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0022.html
sourcePath: /source/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0022.html
indexId: d6ef0ebbd6a0b1dab40c64e12daf19aa2c4b287fc9be3fdbfe94c7adfd19515784
---
# Add自定义算子开发

本入门教程，将会引导你完成以下任务，体验Ascend C SIMT算子开发基本流程。

1. 算子分析，明确数学表达式和计算逻辑等内容；
2. Add算子核函数开发；
3. 算子核函数运行验证。

在正式的开发之前，需要先完成环境准备工作，开发Ascend C算子的基本流程如下图所示：

图1 开发Ascend C算子的基本流程

- 请点击LINK(https://gitcode.com/cann/asc-devkit/tree/9.0.0/examples/03_simt_api/00_introduction/01_add)获取样例代码。
- 使用本教程只需要您具有一定的C/C++基础，在此基础上，如果您已经对Ascend CSIMT编程模型有一定的了解，您可以在实际操作的过程中加深对理论的理解；如果您还没有开始了解Ascend CSIMT编程模型，也无需担心，您可以先尝试跑通教程中的样例，参考教程最后的  指引
进行进一步的学习。


#### 环境准备

- CANN软件安装
  开发算子前，需要先准备好开发环境和运行环境，开发环境和运行环境的介绍和具体的安装步骤可参见《CANN 软件安装(https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html)》。

- 环境变量配置
  安装CANN软件后，使用CANN运行用户进行编译、运行时，需要以CANN运行用户登录环境，执行source ${INSTALL_DIR}/set_env.sh命令设置环境变量。${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。


#### 算子分析

主要分析算子的数学表达式、输入输出的数量、Shape范围以及计算逻辑的实现，明确需要调用的Ascend C SIMT接口或操作符。下文以Add算子为例，介绍具体的分析过程。

1. 明确算子的数学表达式及计算逻辑。
  Add算子的数学表达式为：

  计算逻辑是：逐元素将外部存储Global Memory对应位置上的输入x与y相加，结果存储在Global Memory输出z上。

2. 明确输入和输出。

  - Add算子有两个输入：x与y，输出为z。
  - 本样例中算子输入支持的数据类型为float，算子输出的数据类型与输入数据类型相同。
  - 算子的输入、输出shape为（48，256）。
3. 确定核函数名称和参数。

  - 本样例中核函数命名为add_custom。
  - 根据对算子输入输出的分析，确定核函数有3个输入输出参数x，y，z， 数据类型均为float。
  - 增加一个核函数入参total_length，用于记录算子实际的输入、输出数据长度，数据类型为uint64_t。
4. 确定算子实现逻辑。

  - 将数据均分到48个Thread Block上，每个Thread Block起256个线程处理256个元素，每个线程处理一个元素。
  - 通过每个线程独有的线程索引，计算当前线程需要处理的数据的偏移量。


通过以上分析，得到Ascend CSIMT实现的Add算子的设计规格如下：
**表1 Add算子输入输出规格**

| name | shape | data type | format |
| --- | --- | --- | --- |
| x（输入） | 48 \* 256 | float\* | ND |
| y（输入） | 48 \* 256 | float\* | ND |
| z（输出） | 48 \* 256 | float\* | ND |
| total\_length | \- | uint64\_t | \- |


- 核函数名称：add_custom
- 算子实现文件名称：add.asc


#### 核函数开发

通过当前线程块索引blockIdx、单个线程块包含的线程数blockDim、当前线程索引threadIdx计算获得当前线程的索引，以当前线程索引作为当前计算数据行的偏移量。```
int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
```


通过下标偏移和加法运算符，计算该偏移位置的数据相加的结果，并将结果写入到输出中。```
z[idx] = x[idx] + y[idx];
```


完整的核函数代码实现如下所示：

```
__global__ void add_custom(float* x, float* y, float* z, uint64_t total_length)
{
    // Calculate global thread ID
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Maps to the row index of output tensor
    if (idx >= total_length) {
        return;
    }
    z[idx] = x[idx] + y[idx];
}
```


#### 核函数运行验证

完成Kernel侧核函数开发后，即可编写Host侧的核函数调用程序。实现从Host侧的APP程序调用算子，执行计算过程。

1. Host侧应用程序框架的编写。
  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 //Host调用需要的头文件 #include <vector> #include "acl/acl.h" //核函数开发部分 __global__ void add_custom(float* x, float* y, float* z, uint64_t total_length) { ... } // 通过<<<...>>>内核调用符调用算子 std::vector<float> add(std::vector<float>& x, std::vector<float>& y) { ... // Calc splite params uint32_t block_num = 48; uint32_t thread_num_per_block = 256; uint32_t dyn_ubuf_size = 0; // No need to alloc dynamic memory. // Call kernel funtion with <<<...>>> add_custom<<<block_num, thread_num_per_block, dyn_ubuf_size, stream>>>(x_device, y_device, z_device, x.size()); ... return output; } // 计算结果比对 uint32_t verify_result(std::vector<float>& output, std::vector<float>& golden) { if (std::equal(output.begin(), output.end(), golden.begin())) { std::cout << "[Success] Case accuracy is verification passed." << std::endl; return 0; } else { std::cout << "[Failed] Case accuracy is verification failed!" << std::endl; return 1; } return 0; } // 验证算子主程序 int32_t main(int32_t argc, char* argv[]) { constexpr uint32_t in_shape = 48 * 256; std::vector<float> x(in_shape); std::vector<float> y(in_shape); std::vector<float> golden(in_shape); ... std::vector<float> output = add(x, y); return verify_result(output, golden); }

2. 编写通过<<<...>>>内核调用符调用算子的代码。
  图2 调用步骤

  如下示例中的acl API使用方法请参考《Runtime运行时 API(https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclcppdevg_03_1952.html)》。

  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 std::vector<float> add(std::vector<float>& x, std::vector<float>& y) { size_t total_byte_size =x.size() * sizeof(float); int32_t device_id = 0; aclrtStream stream = nullptr; uint8_t* x_host = reinterpret_cast<uint8_t *>(x.data()); uint8_t* y_host = reinterpret_cast<uint8_t *>(y.data()); uint8_t* z_host = nullptr; float* x_device = nullptr; float* y_device = nullptr; float* z_device = nullptr; // Init aclInit(nullptr); aclrtSetDevice(device_id); aclrtCreateStream(&stream); // Malloc memory in host and device aclrtMallocHost((void **)(&z_host), total_byte_size); aclrtMalloc((void **)&x_device, total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc((void **)&y_device, total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST); aclrtMalloc((void **)&z_device, total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST); aclrtMemcpy(x_device, total_byte_size, x_host, total_byte_size, ACL_MEMCPY_HOST_TO_DEVICE); aclrtMemcpy(y_device, total_byte_size, y_host, total_byte_size, ACL_MEMCPY_HOST_TO_DEVICE); // Calc splite params uint32_t block_num = 48; uint32_t thread_num_per_block = 256; uint32_t dyn_ubuf_size = 0; // No need to alloc dynamic memory. // Call kernel funtion with <<<...>>> add_custom<<<block_num, thread_num_per_block, dyn_ubuf_size, stream>>>(x_device, y_device, z_device, x.size()); aclrtSynchronizeStream(stream); // Copy result from device to host aclrtMemcpy(z_host, total_byte_size, z_device, total_byte_size, ACL_MEMCPY_DEVICE_TO_HOST); std::vector<float> output((float *)z_host, (float *)(z_host + total_byte_size)); // Free memory aclrtFree(x_device); aclrtFree(y_device); aclrtFree(z_device); aclrtFreeHost(z_host); // DeInt aclrtDestroyStream(stream); aclrtResetDevice(device_id); aclFinalize(); return output; }

3. CMake编译配置如下：
  1 2 3 4 5 6 7 8 9 10 11 12 13 14 cmake_minimum_required(VERSION 3.16) # find_package(ASC)是CMake中用于查找和配置Ascend C编译工具链的命令 find_package(ASC REQUIRED) # 指定项目支持的语言包括ASC和CXX，ASC表示支持使用毕昇编译器对Ascend C编程语言进行编译 project(kernel_samples LANGUAGES ASC CXX) add_executable(demo add.asc ) # 通过编译选项设置NPU架构 target_compile_options(demo PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510 --enable-simt> )

4. 编译和运行命令如下：
  1 2 3 mkdir -p build && cd build; cmake ..;make -j; ./demo

  该样例仅支持如下型号： Atlas 350 加速卡 --enable-simt用于指定SIMT编程场景。 --npu-arch用于指定NPU的架构版本，dav-后为架构版本号，请替换为您实际使用的架构版本号。各AI处理器型号对应的架构版本号请通过AI处理器型号和__NPU_ARCH__的对应关系进行查询。


#### 接下来的引导

如果您想了解更多SIMT编程相关概念，可以参考AI Core SIMT编程学习基本概念，再来回顾本教程；如果您已经了解相关概念，并跑通了该样例，您可以参考SIMT算子实现了解Ascend CSIMT编程中的更多细节。
