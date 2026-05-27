# 类人视觉闭环模型

Human-like Closed-Loop Vision Model

## 核心思想

当前视觉模型都是**开环**的：图像 → 特征 → 分类，一次前向通过。
人眼是**闭环**的：预测 → 比较 → 更新 → 再预测，迭代收敛。

本项目从零实现一个带物理世界预测的三层闭环视觉架构。

## 三层架构

```
① 基础视觉层 (BaseVision)
   像素 → 层次化特征图
   ConvNeXt-Tiny, 预训练权重

② 物理预测层 (PhysicalPredictor)
   特征图 → 物理量预测 (遮挡/深度/前景背景)
   这些预测有真实物理意义，不是特征统计关联

③ 状态调节层 (StateManager)
   维护隐状态 → 生成预期物理量 → 比较误差 → 更新状态
   自适应 gain 调度，控制"相信预测 vs 相信实际"的权重
```

## 项目结构

```
├── ARCH.md                     架构设计文档
├── README.md                   本文件
├── requirements.txt            依赖
├── src/
│   ├── backbone.py             ① 基础视觉层
│   ├── physical_predictor.py   ② 物理预测层
│   ├── state_manager.py        ③ 状态调节层
│   ├── closed_loop.py          三层整合
│   └── train.py                训练/评估/对比实验
└── scripts/
    └── generate_depth_pseudo_labels.py  生成深度伪标签
```

## 快速开始

```bash
pip install -r requirements.txt

# 对比实验: Baseline (开环 ConvNeXt) vs Closed-Loop
python -m src.train --data-root /path/to/imagenet
```

## 预期结果

| 条件 | Baseline | Closed-Loop |
|:----|:--------|:----------|
| 干净 ImageNet | 82.1% | ≈ 82.1% (不退化) |
| 噪声/遮挡 | 跌 | 少跌 1-3% |
| 对抗样本 | 崩 | 更鲁棒 |
