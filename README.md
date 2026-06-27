# Essence Field — 物质表达维度 视觉本质场

> **零训练。零黑箱。物质在表达维度上自发显现。**

---

## 核心哲学

物质不需要被检测。物质通过 8 种表达维度（圆度、围合、边缘、纹理方向、曲率、亮度变化、块状、凹陷）在图像中主动显现自己。

8 个表达维度 = 2D 图像中物质能表达自己的**本体属性全集**。不是"模型学到了这些特征"——是物质本身就有这些属性，模型不过是在读取。

## 三层架构

```
输入图像
  ↓
表象层（6维）— 物质如何回应光
  · 亮度对比 · 色相对比 · 纹理粗细 · 边缘密度 · 方向一致 · 高光
  ↓
抽象层（8维）— 物质的内在结构  
  · 圆度 · 围合 · 边缘 · 纹理方向 · 曲率 · 能量 · 块状 · 凹陷
  ↓
本质层 — 表象⊗抽象跨层融合 → 边缘感知扩散 → 流域分割
  · 物质在场中自发沉淀为稳定态
  · 零 K-means，零指定 K
```

## 关键结果

| 验证项 | 结果 |
|--------|------|
| Caltech101 泛化 | 113/113 全部收敛 |
| kun(围合) 光照不变 | CV=2%（亮度 3x 变化） |
| 跨图匹配 | 海豚 0.98, 飞机 0.99 |
| 键帽完整性 | v1.0 已解决（豆包确认） |

## 快速开始

```bash
# 一键物质发现
python scripts/run.py image.jpg

# 二图物质对比
python scripts/compare.py cup1.jpg cup2.jpg

# 批量泛化测试
python scripts/test_generalize.py --dir test_images/

# 理解报告
python scripts/report.py image.jpg
```

## 文件结构

```
src/
  operators.py          — 8 个固定表达维度
  field_core.py         — 本质场计算（三层管线）
  field_dynamics.py     — 场动力学（引力/斥力/摩擦）
  diffusion.py          — 场自组织扩散
  primitive.py          — 基元发现 + 跨图匹配
  material.py           — 物质读出
scripts/
  run.py                — 一键物质发现
  compare.py            — 二图对比
  test_generalize.py    — 批量泛化测试
  report.py             — 理解报告
```

## 分支

| 分支 | 内容 |
|------|------|
| `main` (v0.1) | 零参数基线，已验证 |
| `feat/global-context` (v1.0) | 三层链路，Watershed，当前主开发 |
| `feat/proposal-mode` | 提案模式原型（v2.0 储备） |
| `feat/physical-ops` | 物理算子因果链（参考） |
