# Round 23: foreach_lerp_scalar —— 我的设计（A 节前置）

> 语义：`y_i = x1_i + weight·(x2_i − x1_i)`，两列表 + Host 标量 weight（fp32）。
> 已读：aclnn 文档 + `foreach_lerp_scalar.h` 结构概览（369 行独立类，非工厂派生）。

## A. CHECKLIST 设计必答

1. **API 选型**：lerp = Sub + Axpy 两步（`x2−x1` 后 `Axpy(x1, delta, w)`）；无高阶封装。
2. **数值稳定性分支（侦察发现，实现重点）**：生产按 weight 幅度分支——|w|<1 用 x1 作基；否则代数翻转为 `y = x2 + (w−1)·(x1−x2)`，把 Axpy 系数拉回 [−1,1) 区间，避免大系数放大 delta 的舍入误差。**系数幅度是编译期不可知、运行时可判的分支条件 → kernel 内 if（非 TilingKey，因 weight 是运行时数据）**。
3. **工厂 vs 独立类（R18 结论第二例）**：lerp 数值分支复杂 → 独立类 369 行；简单三目运算才进工厂。
4. **fp16 路径**：cast 双槽 fp32 → Sub/Axpy 全在 fp32 域（weight 精度不降级）→ cast 回。
5. **UB 预算**：x1/x2 两队列（BUFFER_NUM=2）+ fp32 双槽 + 输出队列。

## 1. 我的实现设计

- 独立类 `MyLerpScalar<T>`：Init 读标量（MTE2_S 事件对）；逐张量走表、均匀分段（复用 R16-R20 骨架）；
- Compute（fp32）：幅度分支两形态；fp16：cast→分支→cast；
- weight 的 Host 侧标量按 tiling 下发 vs Device 读——侦察显示 weightVal 是成员 float，Init 阶段读取（走 GM scalar + MTE2_S）。

## 2. //?? 清单

1. FLOAT_NUM_POS/NEG 的精确值（预测 ±1.0）与判定用 `<` 还是 `<=`。
2. weight 读入路径：tiling 直传 float 还是 GM 标量读（侦察见成员 float + Init，倾向 tiling 下发——但 aclnn 是 Device aclScalar？核对）。
3. 大系数分支的代数翻转是否影响 mean/端点权重退化（w=1 时 delta 翻转后系数 0——正确性 OK）。
4. fp16 的 weight 用 fp32 参与（Axpy<T,float>）还是转 T。
