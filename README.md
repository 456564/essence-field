# 本质空间 — 8 物理算子视觉架构

> **不预设概念，只定义存在状态。理解从物理推演中涌现，不由训练数据灌输。**

---

## 一、核心哲学

物质在空间中存在，只能处于 8 种基本状态之一。8 个算子各测一种——不是特征检测器，是物质存在方式的数学呈现。

| 对 | 算子 | 物理量 | 公式 |
|:--|:-----|:------|:-----|
| 动–静 | dong/jing | 变化速率 / 不变 | Sobel / 1-dong |
| 刚–柔 | gang/rou | 硬边界脊线 / 渗透性 | dong-box(dong) / 边柔+面柔 |
| 聚–散 | ju/san | 围合度 / 开放性 | 方向覆盖×距离内部 |
| 阳–阴 | yang/yin | 实体占据 / 虚空 | ju×(1-dong)×(1-gang)×(1-cu) |

**+ cu(纹理能量) + dist(到边欧氏距离)**

0 可训练参数。全部固定数学运算。

---

## 二、架构

```
RGB [B,3,H,W]
  ↓ PhysicalOperatorLayer (8算子, 固定)
8 通道 [B,8,H,W]
  ↓ EssenceSpace (统一封装)
  ↓ build_wall (刚+阳 → 不可穿越墙)
  ↓ ParticleSimulator (粒子随机游走+重力)
  ↓ retention_rate (滞留率)
  
物体 = 粒子被陷住 → 容器
非物体 = 粒子逃逸 → 开放空间
```

---

## 三、粒子推演

无需训练，无需标签。系统通过物理模拟自行发现功能属性：

| 物体 | 滞留率 | 系统发现 |
|:-----|:------|:--------|
| 杯子 | 1.00 | 容器 — 粒子被困在杯腔内 |
| 闭合方框 | 1.00 | 封闭容器 |
| 披萨盒 | 1.00 | 容器 |
| 相机 | 0.00 | 非容器 — 粒子逃逸 |
| 办公椅 | 0.00 | 非容器 |
| 海豚(水面) | 0.62 | 部分围合 |

---

## 四、项目结构

```
Conv2d/
├── ARCH.md                        架构演进文档
├── README.md                      本文件
├── test_maccup.png                测试图片
│
├── src/
│   ├── operators.py               8 个固定物理算子
│   ├── essence_space.py           本质空间封装 + 墙掩码
│   ├── simulation.py              粒子推演引擎
│   ├── pipeline.py                投影+融合流水线
│   └── visualize.py               可视化工具
│
├── scripts/tests/
│   ├── test_single_ops.py         单算子测试 [dong|gang|cu|rou|ju|dist|yang|yin|all]
│   ├── test_yang_multi.py         8列算子对比
│   ├── test_simulation.py         粒子推演
│   ├── test_cu_multi.py           纹理测试
│   ├── test_ju_multi.py           围合测试
│   ├── test_rou_multi.py          渗透测试
│   └── test_eval.py               聚类评估
│
├── archived/                      旧架构代码归档
├── data/                          数据集（不跟踪）
└── test_output/                   输出的可视化图片
```

---

## 五、快速开始

```bash
# 8算子单张测试
python scripts/tests/test_single_ops.py yang

# 8列全对比 + yin overlay
python scripts/tests/test_yang_multi.py

# 粒子推演 — 容器检测
python scripts/tests/test_simulation.py

# 全部4对8算子
python scripts/tests/test_single_ops.py all
```

---

## 六、设计约束

- **不训练算子** — 温度计不准自己调
- **连续场** [0,1] — 8 个算子全是连续值，不是二值标签
- **局部测量** — 不做全局拓扑判断（ju 的方向覆盖是统计，不是连通性）
- **物理诚实** — 2D 图像给不到的信息（杯腔中空 vs 实体），算子不假装知道
- **互补完备** — 四对互补覆盖全空间：阳+阴=1, 动+静=1, 刚+柔=不补但完, 聚+散=1

---

## 七、版本

| 分支 | 内容 |
|:-----|:-----|
| `main` | 旧八卦架构 (archived) |
| `physical-v2` | 8 物理算子 + EssenceSpace + 粒子推演 (当前) |
