# Round 33: IndexSelect/Gather —— 我的设计（A 节前置）

> 语义（aclnnIndexSelect）：out[i][...] = src[index[i]][...]（axis=0 行选）或沿 last-dim 元素选。
> 已读：`gather_v3/` 4 模板变体 + GatherMask 两种用法。

## A. CHECKLIST 设计必答

1. **API 选型**：gather 的核心是 **GatherMask 指令**（两种形态）：
   - **索引模式**：mask 参数是索引 tensor，按索引从 UB 聚集元素（内维 gather 的主力）；
   - **掩码位模式**：mask 是 uint32 位图（CompareScalar+And 生成），bit=1 的元素被**压缩**到输出（rsvdCnt 返回个数）——生产用它做"索引范围过滤+压缩"，避免标量循环。
2. **路径分派**：axis=0 行选（行连续）→ 逐行 DataCopy 即可，**不需要 GatherMask**；内维元素选 → GatherMask 索引模式。生产的 4 模板（tmpl_0..3）按 axis/布局/索引 dtype 分派。
3. **索引预处理**：idx → fp32 → CompareScalar(GE begin, LT end) → And → 位图 → GatherMask 压缩（生产 CalcIdxInRange）——**索引合法性过滤全部 tensor 化**（B3 精神在索引域的延伸）。
4. **UB 预算**：src 行块 + idx + cmp 位图 + 压缩缓冲。

## 1. 我的实现（两条路径 mini）

- pathA（axis=0）：逐行 DataCopy（行连续）；
- pathB（内维 gather）：载入 src 行块 + idx → GatherMask 索引模式逐行聚集（//?? 索引模式的确切参数形态）。
