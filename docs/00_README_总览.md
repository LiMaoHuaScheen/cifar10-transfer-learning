# 第一步 · CIFAR-10 图像分类迁移学习 —— 资料总览

> 考研 AI 方向简历项目 · 方案 A（视觉线）· 第一站
> 项目根目录：F:/Comfy-Desktop/ai-project/

## 一句话概括

在同一份 CIFAR-10 数据集、同一个 ResNet-18 骨干网络上，对比「随机初始化从头训练」与「ImageNet 预训练权重微调」两条路线，用实测数据证明迁移学习的收敛速度与最终精度优势。

**核心结论：71.48% → 82.04%，绝对提升 10.56 个百分点。**

## 本文件夹内容

| 文档 | 内容 | 什么时候看 |
|------|------|-----------|
| 01_方案说明.md | 为什么选这个实验、技术路线、方法论设计 | 想理解整体思路时 |
| 02_环境搭建.md | Python / torch / CUDA / 数据集的完整安装步骤与踩坑记录 | 换机器或重装环境时 |
| 03_操作步骤.md | 从零到出结果的逐步教程（含命令） | 想亲手复现一遍时 |
| 04_实验报告.md | 逐轮数据、结果分析、结论 | 写论文 / 复试准备时 |
| 05_简历与面试.md | 简历写法 + 面试高频问题与参考答案 | 投简历 / 模拟复试时 |
| 06_代码说明.md | train_cifar.py 逐段讲解 | 想读懂或改代码时 |

## 项目目录结构

~~~
F:/Comfy-Desktop/ai-project/
├── train_cifar.py              训练主脚本（双实验对照）
├── plot_curve.py               可视化脚本
├── data/
│   ├── cifar-10-python.tar.gz         数据集压缩包 (162.6 MB)
│   └── cifar-10-batches-py/           解压后的数据集
│       ├── batches.meta
│       ├── data_batch_1 .. data_batch_5   训练集 45000 张
│       └── test_batch                    测试集 10000 张
├── output/
│   ├── result_summary.json      全部实验数据
│   ├── learning_curve.png       学习曲线图
│   └── accuracy_bar.png         准确率对比柱状图
└── docs/step1_cifar10/          本文件夹
~~~

## 环境速查

| 项 | 值 |
|----|----|
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8GB VRAM |
| Python | 3.12.14 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA | 12.8 (driver 621.24, UMD 13.5) |

## 核心复现命令

~~~powershell
cd cifar10-transfer-learning
& python train_cifar.py --data-root data --epochs 10 --device cuda
& python plot_curve.py
~~~

> 注：本机 python 不在 PATH 中，故用绝对路径调用解释器。
