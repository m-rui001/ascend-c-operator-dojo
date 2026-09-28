# Round 21: LayerNorm（single-read 策略）—— 我的设计（CHECKLIST A 节前置）

> 目标：LayerNorm 前向 y = (x−mean)·rstd·γ+β，输出 mean/rstd。variance 路径是本轮主题。
> 已读：`layer_norm_v4_single_read.h` 核心段（126-250 行）。

## A. CHECKLIST 设计必答

1. **API 选型**：归约用 ReduceSum；**variance 路径有两种数学形态**——(a) 两遍 E[(x−mean)²]（再读一遍 x 或缓存全行）(b) 单遍 E[x²]−mean²（读一遍，缓存 scaled x）。生产选 (b)（"single_read"文件名即此意）。
2. **产品矩阵**：fp16/bf16/fp32。
3. **封装边界**：无高阶 LayerNorm API（LayerNorm 是 v2 工厂外的独立类）。
4. **瓶颈**：访存——(b) 少读一遍 x，带宽减半。
5. **精度 vs 带宽**：E[x²]−mean² 数学上不如两遍稳定（相消误差），生产靠 **fp32 中间量**兜住（B8）；验证环节确认。
6. **UB 预算**：x 缓存整块 nRow×rowAlign（fp32）+ y（scaled）+ 输出 fp32 区——**多行一块是前提**（行数 × 列长 ≤ UB）。
7. **多核**：按行块切，行是原子。

## 1. 我的实现设计

- fp32 输入（省 cast 层），x 缓存整块；mean 与 rstd 按**行循环内标量化**：
  - 行归约用我已掌握的 WholeReduceSum；
  - mean = Σx·(1/N)；var = Σx²·(1/N) − mean²；rstd = 1/√(var+eps)（标量数学）
  - y 行 = (x−mean)·rstd 后乘 γ 加 β
- γ/β 的加载时机：我按"行循环前一次载入"简化，对比环节核对生产的"行循环内延迟加载+V_MTE2 事件对"。
- mean/rstd 输出：攒进 UB 再 DataCopyPad 一次性写出（B5 攒批）。

## 2. //?? 清单

1. 生产的 SetMaskCount/SetVectorMask/GetAccVal 累加器读数 idiom——与我用 WholeReduceSum+V_S 的取舍。
2. 生产 coefficient 的精确定义（从 Muls 两次的代数反推：coef 是否 1/N）。
3. γ/β 行循环内加载的调度（预取深度 1 行的流水）。
4. ReinterpretCast 双半区布局（fp32 区 | 原精度区）的通用性。
