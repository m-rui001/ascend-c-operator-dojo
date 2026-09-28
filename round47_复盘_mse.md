# Round 47 复盘：MSELossV2（全量标量归约的跨核两段）

> 我的实现：`my_impl/round47_mse/`（DESIGN.md / mse_loss_custom.h）
> 对比对象：`loss/mse_loss_v2/op_kernel/mse_loss_v2_base.h + mse_loss_v2_sum.h`（ReduceSumBisect/CopyOut）

---

## 一、ReduceSumBisect（R26 折半求和的块对齐变体）

```cpp
while (len > 8) {
    offset = ((len + 15) >> 4) << 3;              // Ceil(Ceil(len,8)/2)*8
    Add(src, src, src[offset], len - offset);     // 尾半折叠到前半
    len = offset;
}
Muls(src, src, scale, len);                        // scale 在收尾一次乘
```

与 R26 `ReduceSumHalfInterval` 的差异：**每次折半强制 8 元素（32B）块对齐**（尾半长度也按块取整），使折叠结果始终 block 对齐，省去收尾前的对齐处理。R26 版本以 2 幂为单位；本版以块为单位——**同族两种对齐策略，按下游需要选**。

## 二、单标量输出的跨核两段（B28 最简实例）

部分和（每核 8 float 槽）→ workspace → `SyncAll()` → **core0 读全部槽 → ReduceSum → cast → DataCopyPad 标量**。要点：
- 槽宽 = 8 float（正好 32B 一块），核数 × 槽 的 workspace 布局；
- `SyncAll()` 在 310p（200 架构）需要 syncLocal 缓冲参入，910b 无参——**架构差异连同步 API 签名都变**；
- 输出 cast（fp32→T）+ `DataCopyExtParams{1, sizeof(T)}` 字节粒度标量写出（B18）。

## 三、mode 即子类（对比 TilingKey 分派）

MSELossV2Base → Sum → Mean：mean 在 sum 之上只改 scale——**模式少且静态时继承叠加**（每层只改一个环节）；模式多或需运行时选择时才用 TilingKey（R24 的 6 变体）。两种模式组织的选择判据补进 CHECKLIST。

## 四、我的自查 bug（B14 再证）

初稿把 y 行 DataCopy 进了 acc 缓冲（应入独立缓冲）——`acc` 是跨段累加器被覆盖。缓冲生命周期表（B14）再次在写实现时拦住真 bug：**diff/平方/累加需要三个独立角色缓冲**（diff 可复用 tmp，y 与 acc 不可共用）。

## 五、CHECKLIST 增量

- **B42（新）**：全量标量输出 = ReduceSumBisect（块对齐折半）+ workspace 每核 8-float 槽 + SyncAll + core0 ReduceSum 聚合；mean 的 1/N 在收尾一次乘；mode 少且静态→子类继承叠加，多/动态→TilingKey。
- **架构差异记录**：SyncAll 310p 带 syncLocal 参、910b 无参——同步 API 签名随架构变化。

## 下一轮候选

activation 目录（silu/gelu_grad）或 50 轮元轮预热（CHECKLIST 去重）。
