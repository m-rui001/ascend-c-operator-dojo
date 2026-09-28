---
title: 环境准备
description: "- 进行Ascend C算子开发之前，需要安装驱动固件和CANN软件包，请参考"
url: https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0003.html
sourcePath: /source/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0003.html
indexId: 4579f14790d1364d7bbc31b50f8707450c70cca48377bf68b78a4890ca537eb884
---
# 环境准备

- 进行Ascend C算子开发之前，需要安装驱动固件和CANN软件包，请参考
《CANN 软件安装(https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html)》完成环境准备。

  1. 安装驱动固件（仅昇腾设备需要），安装步骤请参见“安装NPU驱动和固件”章节。
  2. 安装CANN软件包，可参考“安装CANN”完成快速安装，可参考其他章节了解更多场景的安装步骤。
    安装CANN软件后，使用CANN运行用户进行编译、运行时，需要以CANN运行用户登录环境，执行source ${INSTALL_DIR}/set_env.sh命令设置环境变量。${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

- 安装cmake。通过cmake编译Ascend C算子时，要求安装3.16及以上版本的cmake，如果版本不符合要求，可以参考如下示例安装满足要求的版本。
  示例：安装3.16.0版本的cmake（x86_64架构）。

```
mkdir -p cmake-3.16 && wget -qO- "https://cmake.org/files/v3.16/cmake-3.16.0-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C cmake-3.16
export PATH=`pwd`/cmake-3.16/bin:$PATH
```


对于Ascend C算子的开发，并非必须安装驱动固件。在非昇腾设备上，可以利用CPU仿真环境先行进行算子开发和测试，并在准备就绪后，利用昇腾设备进行加速计算。非昇腾设备的安装请参考《CANN 软件安装(https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html)》中“附录B：常用操作 > 在非昇腾设备上安装CANN”章节。
