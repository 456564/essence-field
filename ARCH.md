# 物理本质空间 — 8物理算子视觉架构

## 架构转向（2026-05-30）

旧八卦语义算子（乾=圆、坤=平...）是人为解释，不是物理测量。新架构：8 算子测量可验证的物理量，理解从物理推演产生。

## 算子依赖链

```
dong(梯度)
  ├→ cu(纹理) — 同层，dong 正相关
  ├→ jing = 1-dong
gang(硬边界) = dong - box_filter(dong) 多尺度
  ├→ dist(距边) = OpenCV 距离变换
  ├→ rou(软纹理) = dong×gang + (1-dong)×(1-cu)
ju(围合) = 多方向 max-pooling 找对面边缘
  │         × sigmoid(dist-0.08) × (1-gang_density×10)
  ├→ san = 1-ju
  ├→ yang(实体) = ju × (1-dong) × (1-gang) × (1-cu)
  ├→ yin = 1-yang
void_prob(虚空概率) = ju × (1-cu) × jing × (1-yang) × (gang<0.01)

每个算子都是纯函数，无可训练参数。
```

## 实现状态

| 算子 | 物理量 | 依赖 | 状态 |
|:-----|:------|:-----|:-----|
| dong | 梯度幅值 | RGB | ✅ |
| cu | 纹理凸起 | RGB+dong | ✅ |
| gang | 硬边界 | dong | ✅ |
| rou | 软纹理 | dong+gang+cu | ✅ |
| dist | 距边距离 | gang | ✅ |
| ju | 围合 | gang+dist | ✅ |
| san | 开放(1-ju) | — | ✅ |
| yang | 实体 | dong+gang+cu+ju | ✅ |
| yin | 虚空(1-yang) | — | ✅ |
| void_prob | 虚空概率 | dong+gang+cu+ju | ✅ |

## 流水线（PhysicalPipeline v3）

```
RGB [B,3,H,W]
  ↓ PhysicalOperatorLayer (9通道: 8算子 + void_prob)
8 响应图 [B,8,H,W]   (void_prob 跳过)
  ↓ 1×1 conv 投影 (1→8 per channel)
8×8 基 [B,8,8,H,W]
  ↓ BilinearFusion (W_up × W_dn, 双投影点积)
64 维交互场 [B,64,H,W]
  ↓ + 空间坐标 (y,x, 归一化到[-1,1])
66 维本质场 [B,66,H,W]
```

## 设计约束

- 所有算子输出非负，值域 [0,1]
- 投影权重、W_up、W_dn clamp ≥ 0
- 下层不依赖上层（dong 不读 gang）

## 当前状态

- ✅ 全部 8 个物理算子已实现并单算子验证通过
- ✅ PhysicalPipeline v3 (BilinearFusion W_up/W_dn + 空间坐标)
- ✅ EssenceSpace (wall_mask, cavity_mask)
- ✅ ParticleSimulator (粒子推演 + 滞留率)
- ✅ camera_demo.py (实时摄像头粒子推演)
- ✅ train_contrastive.py (对比学习 v2)
- ✅ test_single_ops.py (合成图 + 实拍单算子验证)
- ⬜ visualize.py 待更新匹配新算子名
- ❌ 跨图片物质类型匹配（未开始）
- ❌ A核谱分解（未开始）

## 文件结构

```
src/
  __init__.py         ← 导出算子 + PhysicalPipeline
  operators.py        ← 8物理算子 + PhysicalOperatorLayer (9通道)
  pipeline.py         ← PhysicalPipeline v3 (66维本质场)
  essence_space.py    ← EssenceSpace (wall/cavity/void)
  simulation.py       ← ParticleSimulator (粒子推演)
  visualize.py        ← ❌ 仍用旧卦象名，待更新
scripts/
  train_contrastive.py    ← 对比学习 v2 (RGB增强 + MLP投影头)
  camera_demo.py          ← 实时摄像头粒子推演
  generate_synthetic.py   ← 合成图生成
  tests/
    test_single_ops.py    ← 每张图 + 合成图单算子验证
    test_eval.py          ← 整体评估 (silhouette)
    test_zero_cluster.py  ← 零训练聚类
    test_simulation.py    ← 粒子推演验证
archived/
  bagua_semantic/         ← 旧八卦架构全部代码
```

## 版本切换

旧八卦架构代码已全部归档至 `archived/bagua_semantic/`。不再兼容。
