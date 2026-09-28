# Round 12: Gelu 算子 —— DESIGN（CHECKLIST 逐项过）

## A. 设计必答

1. **API 选型**：✅ 已确认 **Gelu 高阶 API 存在**（激活函数类，ascendc-api-adv 库；另有 FasterGelu/FasterGeluV2、Erf/Tanh 基础接口可拼）。用 `Gelu(dst, src, count)`（//?? 精确签名/approximate 模式/是否配套 GetGeluTmpSize 待对比销案）。
2. **产品支持矩阵**：//?? 高阶 API 的支持范围待查（生产 gelu_quant 覆盖 训练/推理/A2 全系，推测 OK）。
3. **封装边界**：elementwise 单输入，API 按片（count）处理，核间与搬运输出仍是我的活。
4. **瓶颈判定**：erf/tanh 级超越函数计算量大，**V 可能是瓶颈**（不同于 Add 的 MTE 瓶颈）→ BUFFER_NUM=2 即可，V 慢时加深无益。
5. **融合/物化**：单输入无此决策。
6. **原子/同步**：无。
7. **UB 预算**：x 队列 2 份×BUFFER_NUM + y 队列 2 份×BUFFER_NUM = 4 份；fp32 中间量由 API 内部处理（//?? isReuseSource 语义）。

## B. kernel 必检（实现时落实）

1. 核间四件：32B 块均分 + 大小核 + host 计算 blockDim（模式A）+ workspace=0 ✓
2. DataCopy 对齐：中间片 32B 锚定；**最后核最后一块**字节精确收尾（Round 10 必检项，第 11 轮又漏了，本轮必须落实 DataCopyPad/ExtParams）✓
3. 标量通路：无归约，N/A ✓
4. 2D 块：一维 elementwise，N/A ✓
5. 内存工具箱：仅双队列 ✓
6. L2 hint：x/y 均流过 → DISABLE（逐 tensor 决策）✓
7. BUFFER_NUM=2（V 瓶颈型，加深无益）✓
8. 精度：Gelu 高阶 API 内部 fp32 中间量（//??）；approximate 模式（erf 精确/tanh 近似）→ **用 TilingKey 路由**（C4 呼应）✓

## C. host 必检

1. 三元组：平台 UB/核数 + shape + dtype 长度表 ✓
2. 32B 锚点切分 ✓
3. attr：approximate(string) → TilingKey；无标量系数 ✓
4. TilingKey 路由 approximate 模式 ✓
5. host 可算标量：无 ✓

## D. 流程纪律

- //?? 清单：① Gelu 签名/approximate 枚举 ② 配套 TmpSize 函数与 isReuseSource ③ 产品支持 ④ 生产 gelu_quant 用高阶 API 还是手拼 Erf——对比环节销案。
