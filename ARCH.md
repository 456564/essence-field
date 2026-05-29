# 八卦→64卦 视觉架构

## 核心数据结构

**64 维卦象场 [B, 64, H, W]**

- 每个像素携带 64 维向量 = 该位置"物质本质"的描述
- 同一物质的像素 → 64 维向量相似
- 不同物质的像素 → 64 维向量不同
- 64 维向量不是图片级别的描述，是像素级别的描述

**不要取全图均值。取均值 = 丢弃空间信息 = 背离架构核心。**

## 算子版本演变

| 版本 | 文件 | 颜色处理 | 训练后分离度 | 通过率 | 物体完整度 |
|:----|:----|:--------|:----------|:-----|:---------|
| 灰度版 (gray) | `operators_gray.py` | x.mean(dim=1) 丢弃颜色 | ~10 | <50% | ❌ 大量空洞 |
| RGB版 (rgb) | `operators_rgb.py` | 三通道分别跑算子取max | ~2.3 | ~58% | ❌ 空洞更多 |
| 颜色调制版 (colormod) | `operators_colormod.py` | 可学习1×1颜色卷积(3→1) | ~3.1 | 58% | ⚠️ 80%空洞解决 |
| 颜色原生版 (native) | `operators_native.py` | RGB全通道梯度，无选择性 | ~4.2 | 58% | ❌ 噪声叠加 |
| **固定颜色+原始强度** (fixedcolor) | **`operators_fixedcolor.py`** | **固定颜色投影+原始强度+instancenorm** | **~15.6** | **100%** | **✅ 全部完整** |

当前最佳：**固定颜色版 (fixedcolor)**

切换命令：`python scripts/switch_operators.py [gray|rgb|colormod]`

### 颜色调制版（当前最佳）

每个算子前加一个可学习的 1×1 卷积（3通道→1通道），共 8×(3+1)=32 个参数。
初始颜色偏好：

| 卦 | 初始权重 | 偏好的颜色 |
|:--|:--------|:---------|
| 乾(圆) | [1, 0, 0] | 红 |
| 坤(平坦) | [0, 1, 0] | 绿 |
| 震(边缘) | [0, 0, 1] | 蓝 |
| 巽(纹理) | [1, 1, 0] | 黄 |
| 坎(曲率) | [0, 1, 1] | 青 |
| 离(能量) | [1, 0, 1] | 紫 |
| 艮(块状) | [1, 0.5, 0] | 橙 |
| 兑(凹陷) | [0.5, 0, 1] | 紫蓝 |

算子本身保持灰度逻辑不变。颜色调制的作用：

- 用颜色一致性压制背景噪声（背景通道高度统一）
- 物体颜色区域由对应的颜色通道主导
- 每个算子只对特定颜色的形状敏感 → 稀疏激活

## 8 算子（固定数学运算）

| 卦 | 算子 | 检测目标 |
|:--|:----|:--------|
| 乾 | 圆对称性 | 圆形/全局结构 |
| 坤 | 局部平坦度 | 平坦/背景/承载面 |
| 震 | 拉普拉斯 | 边缘/突变 |
| 巽 | 方向梯度一致性 | 细长纹理/方向性蔓延 |
| 坎 | 等照度线曲率 | 弯曲/流动 |
| 离 | 梯度幅值 | 视觉能量（亮度+饱和度独立加） |
| 艮 | 局部纹理不连续性 | 块状/遮挡边界 |
| 兑 | 中心-周围对比度 | 凹陷/缺口 |

每个算子输出 [B, 1, H, W]。

## 设计约束：全管线非负

**八卦组合成六十四卦 = 正向百分比叠加，不涉减法。**

物理意义：像素的"本质签名" = 各卦响应的正向加权组合。非负 = 每个卦对该像素的贡献 ≥ 0，不存在"负的乾"。

约束：

1. **算子输出非负** ✅ 全部算子用 abs/max/clamp(min=0) 实现，天然满足
2. **投影权重非负** — 1×1 conv 权重 clamp ≥ 0，保证"卦→面相"映射不翻转极性
3. **A 核非负** — 8×8 交互矩阵 clamp ≥ 0，保证卦间组合 = 正向百分比叠加
4. **归一化不产生负值** — InstanceNorm（零均值→半正半负）必须替换

违反后果：训练后 field[8:16]（坤维度）中杯内 < 背景，卦响应方向丢失。

## 双线性融合

```
8 算子响应 [B, 1, H, W] × 8   （全部非负）
    ↓ 1×1 conv 投影到 8 维        （权重 ≥ 0, 无bias）
8×8 特征矩阵 [B, 8, 8, H, W]   （全部非负）
    ↓ 双线性核 A ∈ R^(8×8)       （A ≥ 0, 对角占优初始化）
64 维卦象场 [B, 64, H, W]        （全部非负）
```

A 核对角占优 = 卦的自交互优先，跨卦混合为辅助。A[i,i] 初始较大（~1），A[i,j≠i] 初始很小（~0.02）。

64 维排列：field[i*8 : i*8+8] = 卦i 与各卦的交互。该 8 维的 L2 范数 = 卦i 在图像中的直观强度图。

## 64 维本质空间的两层信息（已验证）

每个 64 维向量包含两层独立信息：

### 第一层：向量范数 → 物体位置（活跃度）
- 背景范数：低且均匀
- 物体范数：高
- 两者分离（训练后更明显），可分割物体
- 范数与表面曲率正相关
- 不区分亮暗

### 第二层：向量方向 → 物质类型
归一化后方向编码物质身份：
- 亮面、阴影、边缘各有独特签名
- 同种物质在不同位置签名一致

## 训练

### 自举训练（bootstrap）

每张图独立：
1. 64维场 → 范数中位数 → 物体/背景掩码
2. InfoNCE损失：物体内像素方向对齐，物体与背景方向分离

可训练参数（固定颜色版）：
- A核：64 参数（双线性融合）
- 投影层：8×(1×1 conv) = 72 参数
- 颜色卷积层：0 参数（颜色权重固定为 register_buffer）
- 总计：~136 参数

训练后效果：
- 固定颜色版分离度~15，100%通过率
- 线性分类 44.7%，显著性MAE 0.15
- 杯内掩码填充38%（坤容器升级中）

## 当前状态

- ✅ 5个算子版本 (gray/rgb/colormod/native/fixedcolor)
- ✅ 固定颜色版算子（八卦颜色投影 + 原始强度 + instance norm）
- ✅ 双线性融合生成64维场
- ✅ 自举训练（InfoNCE + 早停 + 余弦退火 + tqdm）
- ✅ 向量范数→物体定位（分离度~15, 100%通过率）
- ✅ 线性分类探测（Caltech101 44.7%，随机基线~1%）
- ✅ 显著性检测（DUTS-TE MAE=0.15）
- ✅ 声纳可视化面板（`src/visualize.py`）
- ✅ 训练复合图自动保存（每5轮 epochXX_composite.png）
- ⚠️ 坤升级为容器检测器（迭代中 — 已定位根因：管线负值违反设计约束，规划修复中）
- ❌ 跨图片物质类型匹配
- ❌ 多物体场景
- ❌ 基本单元压缩
- ❌ A核谱分解 + 算子颜色雷达图

### 坤容器检测升级（进行中）

核心改动：`_kun` 不再测平坦度(1/方差)，改为"被边缘围合的内部"：
- Sobel边缘 → 软边界 → 51×51大核镜像扩散 → ×颜色一致性
- 原始输出正确（杯内0.924 > 背景0.760）
- 训练投影层后颠倒（杯内 < 背景）→ 根因：全管线存在负值路径
- InstanceNorm 零均值化 + 投影权重可负 + A核可负 → 信号被翻转
- 修复方向：全管线非负约束（见上方"设计约束"）

调试工具：
- `python scripts/test_kun_diag.py` — 对比坤原始输出 vs 投影后（已修复维度索引bug）
- `python scripts/test_kun_maccup.py` — 杯子图坤响应 + 叠加
- `python scripts/test_single_trained.py` — 含argmax复合图

## 版本切换

```bash
# 查看当前版本
python scripts/switch_operators.py list

# 切换版本
python scripts/switch_operators.py gray       # 灰度版
python scripts/switch_operators.py rgb        # RGB版（三通道取max）
python scripts/switch_operators.py colormod   # 颜色调制版
python scripts/switch_operators.py native     # 颜色原生版
python scripts/switch_operators.py fixedcolor # 固定颜色版（推荐）
```

## 文件结构

```
src/
  operators.py              ← 当前算子（固定颜色版副本）
  operators_fixedcolor.py   [VERSION=fixedcolor] 当前最优
  operators_gray.py         [VERSION=gray]
  operators_rgb.py          [VERSION=rgb]
  operators_colormod.py     [VERSION=colormod]
  operators_native.py       [VERSION=native]
  pipeline.py               ← 流水线：算子→投影→融合→64维场
  visualize.py              ← 声纳面板+argmax复合+blended混合
checkpoints_fixedcolor/     ← 当前训练权重
  bootstrap_epoch15.pth     ← 最新（Kun ×10, 15轮）
scripts/
  train_bootstrap.py        ← 自举训练(tqdm+早停+余弦退火)
  switch_operators.py       ← 5版本切换
  test_single_trained.py    ← 6图单张测试(含argmax复合)
  test_generalization_trained.py ← 通用性测试
  test_kun_diag.py          ← 坤诊断(原始vs投影后)
  test_kun_maccup.py        ← 杯子图坤响应
  test_sonar.py             ← 声纳面板
  saliency_test.py          ← DUTS-TE显著性
  linear_probe.py           ← 线性分类探测
