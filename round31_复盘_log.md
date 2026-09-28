# Round 31 复盘：foreach_log（超越函数的极简工厂形态）

> 我的实现：`my_impl/round31_log/`（DESIGN.md / foreach_log_custom.cpp）
> 对比对象：`foreach/foreach_log/op_kernel/foreach_log.cpp`（全文件 ~40 行）

---

## 一、生产形态：一行 Adapter 的完整算子

```cpp
template <typename T>
__aicore__ void LogAdapter(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& u) {
    Log<T>(dst, src);   // 基础 API 指令封装
}
// 实例化：
ForeachImplictOutput<half, half, LogAdapter<half>, 2, 1> op;
```

整个算子 = **一行 Adapter + 三次模板实例化**。R16"算子工厂"的极致案例：当运算恰好映射到一条基础指令（Log 硬件对数指令），算子开发收缩为"选工厂类 + 填函数"。

## 二、两个模板签名新知

1. **进出 dtype 分离**：`ForeachImplictOutput<T_in, T_out, Adapter<T_compute>, ...>`——bf16 实例化为 `<bfloat16_t, float, LogAdapter<float>>`：基类负责 bf16→fp32 cast 进、Adapter 在 fp32 域算、基类 cast 回 bf16 出。**Adapter 永远工作在计算精度域，精度边界归基类**（B8 精度路径的工厂化落实）。
2. **paramsCount=1 是最小值**（R20 addcdiv 为 3）——槽位数 = 算术所需 fp32 中间量个数，按需声明。

## 三、R12 教训的三级递进修正

R12 的"超越函数 = 钳位 + 多项式/exp tensor 化逼近"需要边界限定：

| 层级 | 适用 | 例 |
| --- | --- | --- |
| 高阶 API | 完整算子级封装 | SoftMax（R2） |
| 基础指令 API | 单函数硬件指令封装：Log/Ln/Exp/Sqrt/Reciprocal/Rsqrt | **Log（本轮）** |
| 手拼多项式 | 无任何现成 API 的复合函数 | Gelu（R12） |

**选型必须走满三级**，Gelu 的教训（当时应先查 Ln/Exp 是否可组合出 Gelu）在 foreach_log 得到正面印证。

## 四、CHECKLIST 增量

- **A1 再精化（R31）**：超越函数选型三级递进（高阶 API → 基础指令 API → 手拼多项式），Log/Ln/Exp/Sqrt/Reciprocal/Rsqrt 均有基础指令封装。
- **E9（新）**：`ForeachImplictOutput<T_in, T_out, Adapter, bufferNum, paramsCount>`——进出 dtype 分离，Adapter 工作在计算精度域；paramsCount = fp32 中间量槽数按需声明。

## 下一轮候选

add_layer_norm_quant（双 smooth 量化，norm 族收尾）或 foreach_expm1（组合超越：e^x−1，验证基础 API 组合）。
