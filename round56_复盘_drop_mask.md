# Round 56 复盘：Dropout 位掩码解包（压缩的对偶 + Adapter 类模式）

> 我的实现：`my_impl/round56_drop_mask/`（DESIGN.md / drop_mask_unpack_custom.h）
> 对比对象：`cann-ops-adv/.../flash_attention_score_drop_mask_adapter.h`（184 行；random 目录为空壳换源）

---

## 一、位掩码的双向对偶（R44 对偶闭合）

| 方向 | 原语链 | 用途 |
| --- | --- | --- |
| **压缩**（R44） | CompareScalar → 位图 → bit-packed GatherMask → rsvdCnt | mask 选出元素（输出长度动态） |
| **解包**（本轮） | 位图进 UB → **Select**（src0=全 1 常量，src1RepStride=0 反复读位图，dstRepStride=8 展开）→ Cast bool | 位掩码还原为逐元素 bool（喂给 Mul/乘法掩码） |

生产的空间账本注释（x 元素 = GM x/8 + UB 2x×2 + GM x 字节）：**位存储省 GM 带宽 8 倍，代价是 UB 空间与转换指令**——大算子里位掩码是标准存储形态。

## 二、Adapter 类模式（组织形态第三种）

`FlashAttentionScoreDropMaskAdapter`：独立头、自持队列/SyncAllCores、主算子构造调用——**算子附件类**。与工厂（复用计算骨架）、独立类（退化流水）并列：

| 形态 | 适用 |
| --- | --- |
| 工厂（v2） | 标准读-算-写族 |
| 独立类 | 数值特化/退化流水 |
| **Adapter 附件类（本轮）** | 可选功能的横切模块（dropout/pse/padding），主算子按 hasXxx 编译期旗标挂载 |

## 三、多核细节

- `multiCoreFactorSize`（tiling）切核 + 单核空段 `SyncAllCores() return` 早退；
- 非对齐分支才需要全核同步——**同步跟着数据依赖走**（无依赖分支零同步）。

## 四、CHECKLIST 增量

- **B48（新）**：位掩码解包 = Select（src0 常量 src1RepStride=0 反复读位图，dstRepStride=8 展开）→ Cast bool；位存储省 GM 带宽 8 倍付 UB+转换指令；压缩（R44）与解包（本轮）是位掩码双向原语。
- **E11（新）**：组织形态第三种——Adapter 附件类（可选功能横切模块，独立头自持队列，主算子按 hasXxx 旗标挂载）。

## 下一轮候选

asc-devkit examples 对照（basic_api 现代版库）或 top_k div/mod 复用练习；或 conv 目录（Cube 卷积布局 NCHW→NC1HWC0）。
