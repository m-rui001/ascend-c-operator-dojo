# Round 28: AddRmsNormDynamicQuant —— 我的设计（A 节前置）

> 语义（5 输出）：y1Out=int8 量化 Norm 结果、scale1Out=row_max(|y|)/127、y2Out/xOut/scale2Out（smooth 可选路）。
> 公式：`scale1Out = row_max(abs(y))/127`，`y1Out = round(y/scale1Out)`。
> 已读：aclnn 文档 + helper.h 的 ReduceMaxFP32 / ReduceMaxInplace（带性能注释）。

## A. CHECKLIST 设计必答

1. **API 选型**：行 max = Abs → ReduceMax（mask-count 慢路径 vs **strided in-place Max 折叠快路径**，生产注释"very very slow"/"about 20us faster"——直接抄快路径）；量化 = 标量除法 `scale=127/max` 逐行一次 + `Muls(1/scale)` 张量广播 + Cast int8。
2. **数值契约链（R27 的延续）**：y1Out(int8) 与 scale1Out(fp32) 是**量化输出对**——反量化 y = y1×scale。int8 截断点是融合链的第五个舍入点（xOut T 精度 → y fp32→int8）。
3. **UB 预算**：y 块驻留（要两遍：算 max 再除）+ abs 临时 + scale 攒批小队列。
4. **两遍可行性**：y 驻留 UB（R22 结论：驻留→多遍免费），max 与除法都在片上完成，GM 只读一遍。

## 1. 我的实现设计（量化块 mini）

```
块驻留 y(fp32):
  Abs(v, y)
  rowMax = StridedMaxTree(v)     // in-place 折叠
  scale  = 127/rowMax            // 标量除法每行一次
  Muls(y, y, 1/scale)            // 广播除法
  Cast(yInt8, y, CAST_RINT)      // int8 写出
  scaleOut 攒批写出（fp32）
```
