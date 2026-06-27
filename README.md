# Essence Field — 从最小基元理解物质世界

> **不分类。不训练。不标数据。一个规则 → 场演化 → 物质自涌现。**

---

## 核心哲学

世界不是标签的集合。世界是最小单元的交互产物。

```
沙子(SiO₂) → 风+干燥 = 沙漠
沙子(SiO₂) → 高温+冷却 = 玻璃  
沙子(SiO₂) → 湿+塑+烧 = 陶瓷

同一个基元。不同条件。三种涌现。
```

视觉世界也一样：**邻域相似的像素互相吸引 → 场演化 → 稳定态 = 物质。** 不定义"什么是容器"——容器是场自然汇聚的产物。

---

## 架构

```
输入: RGB 图像
  ↓
唯一规则: 邻域相似 → 吸引, 邻域相异 → 排斥
  ↓
场弛豫: N 轮迭代, 像素在相似邻居的拉力下演化
  ↓
稳定态: 向量一致的连通区域 = 物质域
  ↓
输出: 域掩码 + 场梯度(自然边界) + 不确定区(场未收敛处)
```

**零算子。零分类器。零训练。零预定义。**

## 四条完备交互法则

| # | 规则 | 物理类比 | 效果 |
|---|------|----------|------|
| 1 | **吸引** | 同质相吸 | 相似→靠拢 |
| 2 | **排斥** | Pauli 排斥 | 不相似→推开, 边界锐化 |
| 3 | **多尺度** | 能标重整化 | 纹理不碎, 轮廓保持 |
| 4 | **惯性** | 质量抵抗变化 | 先稳定区域不被扰动 |

## 快速开始

```bash
# Primal Field — 单一规则物质涌现
python scripts/run_primal.py photo.jpg

# 四条规则分别验证
python scripts/run_primal.py photo.jpg --rule 1          # 纯吸引
python scripts/run_primal.py photo.jpg --rule 2 --repulsion 0.1  # +排斥
python scripts/run_primal.py photo.jpg --rule 3          # +多尺度
python scripts/run_primal.py photo.jpg --rule 4          # +惯性

# 跨图物质对比（8算子版）
python scripts/compare.py cup1.jpg cup2.jpg
```

## 已验证

| 验证项 | 结果 |
|--------|------|
| 零参数管线 113 类泛化 | Caltech101 全部收敛 |
| 核心算子光照不变 | kun CV=2%（亮度 3x 变化） |
| 跨图匹配 | 几何一致类 0.97+（海豚/飞机） |
| Primal 单规则涌现 | 杯图自然分出 5 个物质域 |

## 项目结构

```
src/
  primal.py              ← Primal Field（单一规则场）
  operators.py           ← 8 八卦算子（fixedcolor 版）
  field_core.py          ← 零参数本质场（表象⊗抽象融合）
  diffusion.py           ← 边感知扩散
  primitive.py           ← 基元发现(K-means/Hungarian)
scripts/
  run_primal.py          ← Primal 一键运行
  run.py                 ← 融合管线一键
  compare.py             ← 二图物质对比
```

## 分支

| 分支 | 内容 |
|------|------|
| `main` | v0.1 零参数基线 |
| `feat/primal-field` | Primal Field（四条规则） |
| `feat/global-context` | 融合管线实验 |
| `feat/physical-ops` | 物理算子路线（参考） |
