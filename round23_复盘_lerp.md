# Round 23 复盘：foreach_lerp_scalar（数值稳定 base-swap）

> 我的实现：`my_impl/round23_lerp/`（DESIGN.md / foreach_lerp_scalar_custom.h）
> 对比对象：`foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.h`（369 行独立类）
> 本轮主题：**运行时系数幅度分支**——数值稳定性在 kernel 内的落地形态。

---

## 一、base-swap 分支（本轮核心新知，CHECKLIST B26）

lerp `y = x1 + w·(x2−x1)` 生产分两形态：

```cpp
if (weightVal < 0.5f && weightVal > -0.5f) {      // |w| < 0.5
    Sub(x2, x2, x1);  Axpy(x1, x2, w);            // x1 基：x1 += w·(x2−x1)
} else {                                           // 否则翻转
    Sub(x1, x2, x1);  w -= 1.0f;
    Axpy(x2, x1, w);                              // x2 基：x2 += (w−1)·(x1−x2)
}
```

三个要点：

1. **代数恒等翻转**：`x1 + w·(x2−x1) = x2 + (w−1)·(x1−x2)`，翻转后 Axpy 系数绝对值缩小 1；
2. **阈值是 0.5 不是 1.0**（我预测 ±1.0 错）——保证系数落在 (−0.5, 0.5)，Axpy 乘加的舍入放大最小；
3. **分支在 kernel 内运行时做**（weight 是 Device 标量数据，非编译期常量）——与 TilingKey 的编译期分支（R13 B11）构成"分支下沉决策表"：shape/ dtype → TilingKey；数据相关 → kernel 内 if。

## 二、其他确认

1. **工厂 vs 独立类第二例**（R18 后）：lerp 数值分支复杂 → 独立类；印证"特化算子脱框"判断。
2. **fp16 路径 weight 全程 fp32**：双槽 cast → fp32 域 Sub/Axpy → cast 回——Axpy 跨精度系数 `Axpy(half_tensor, half_tensor, float_w)`（R20 发现的跨精度系数用法）。
3. **隐式输出风格**：生产 InnerComputer 后结果留在 x1Local（无独立 outLocal），与 R20 ImplictOutput 同型——lerp 虽有独立 y，中间计算仍原地做。
4. 生产每步之间 PipeBarrier 成对出现（Sub→Axpy 跨缓冲依赖），R20 B21 判据一致。

## 三、我的差距

1. 阈值错了（1.0 vs 0.5）——数值 idiom 的参数要从生产抄，不要凭直觉；
2. 我的 fp32 分支加了 `DataCopy(out, x1)` UB 内拷贝——生产不拷（结果留 x1Local，外层直接写 y，隐式输出），多一次 UB→UB；
3. 我的 fp16 分支未做 base-swap（只在 fp32 域直接算）——生产 InnerComputer 对 fp16 也走同一分支逻辑（cast 后分支）。

## 四、CHECKLIST 增量

- **B26（新）**：插值/加权类算子按系数幅度做 base-swap（阈值 0.5），翻转代数恒等式把 Axpy 系数拉进最小舍入区间；分支在 kernel 内运行时做（数据相关分支不进 TilingKey）。

## 下一轮候选

foreach_unary_v2（纯搬运 v2 工厂）或返回 norm 系列：cann-ops `norm/rms_norm_grad`（反向传播，导数链的 kernel 化）。
