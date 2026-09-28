# Round 31: foreach_log —— 我的设计（A 节前置）

> 语义：y_i = log(x_i)，单列表一元超越函数。
> 已读：`foreach_log.cpp`（极简）——`ForeachImplictOutput<T_in, T_out, Adapter, 2, 1>` + 一行 Adapter 调基础 API `Log<T>`。

## A. CHECKLIST 设计必答

1. **API 选型（R12 精化的正面案例）**：log 有基础 API 指令封装（`Log<T>`/`Ln` 族），Adapter 一行；R12 手拼多项式只适用于**无基础 API 的复合函数**（Gelu）。超越函数选型链 = 高阶 API → 基础指令 API → 手拼多项式，三级递进。
2. **模板签名**：`ForeachImplictOutput<T_in, T_out, Adapter, bufferNum, paramsCount>`——**进出 dtype 分离**（bf16 进、float 域算、bf16 出），cast 边界由基类 InnerComputer 承担，Adapter 只写核心数学（fp32 域）。
3. **paramsCount=1**：无额外 cast 槽需求的最小值（与 R20 的 3 对比——槽位数由算术需要的中间量个数决定）。

## 1. 我的实现

- `MyLogAdapter`：一行 `Log(dst, src)`；
- TilingKey 1/2/4 三 dtype 实例化（bf16 用 float Adapter + 基类回转）。
