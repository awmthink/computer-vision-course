# Hyena

## 概述

### 什么是 Hyena
虽然 Transformer 是一种成熟且非常强大的架构，但其二次计算成本在推理中代价高昂。

Hyena 是一种新的算子，用来替代注意力机制。由 Hazy Research 开发，它具有次二次计算效率，通过隐式参数化的长卷积和数据控制的门控机制交替构建。

<Tip>

长卷积类似于标准卷积，但其核的大小与输入一致，等同于全局感受野，而不是局部的。隐式参数化卷积意味着卷积滤波器的值不是直接学习的，而是通过学习一个可以恢复这些值的函数。

</Tip>

<Tip>

门控机制控制网络中信息流的路径，帮助确定信息应被记住的时长。通常，它们由逐元素相乘组成。
关于门控机制的有趣博客文章可以在 [这里](https://medium.com/autonomous-agents/a-math-deep-dive-on-gating-in-neural-architectures-b49775810dde) 找到。

</Tip>

![transformer2hyena.png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/outlook_hyena_images/transformer2hyena.png)
Hyena 算子通过递归地逐次计算卷积和乘法逐元素门控操作，直到所有投影用完为止。这种方法基于同一研究小组开发的 [Hungry Hungry Hippo (H3)](https://arxiv.org/abs/2212.14052) 机制。H3 机制通过其数据控制的参数分解，作为一种替代注意力的机制。

另一种理解 Hyena 的方式是将其视为 H3 层对任意投影数量的推广，Hyena 层通过不同的长卷积参数化递归地扩展 H3。
![hyena_recurence.png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/outlook_hyena_images/hyena_recurence.png)

### 从注意力到 Hyena 算子

注意力机制有两个基本属性：
1. 它具有全局上下文感知，能够评估序列中视觉标记对之间的交互。
2. 它依赖于数据，意味着注意力方程的操作会基于输入数据本身变化，尤其是输入投影 $q$、$k$、$v$。

![Alt text](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/outlook_hyena_images/self-attention-schema.png)

注意力机制通过三个投影定义：查询 $q$、键 $k$、值 $v$，这些投影是通过将输入视觉标记与三个在训练中学习的矩阵 $W_q$、$W_k$ 和 $W_v$ 相乘生成的。

对于给定的视觉标记，我们可以使用这些投影计算注意力得分。注意力得分决定了对输入图像其他部分的关注程度。
详细的注意力机制解释可以参考这个[图文博客](https://jalammar.github.io/illustrated-transformer/)。

为了复制这些特性，Hyena 算子整合了两个关键要素：
1. 它采用长卷积提供全局上下文感知，类似于注意力机制的第一个属性。
2. 为了实现数据依赖，Hyena 使用逐元素门控。这本质上是输入投影的逐元素相乘，类似于传统注意力的数据依赖特性。

在计算效率方面，Hyena 算子的评估时间复杂度为 $O(L \times \log_2 L)$，表明了处理速度的显著提升。

### Hyena 算子

我们来深入了解 Hyena 算子的二阶递归，以简化表示。
![hyena_mechanism.png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/outlook_hyena_images/hyena-order2-schema.png)

在该阶次下，我们计算 3 个投影，类似于注意力机制中的 $q$、$k$ 和 $v$ 注意力向量。

然而，与典型的使用单一密集层将输入序列投影到表示的注意力机制不同，Hyena 同时包含了一个密集层和在每个通道上执行的标准卷积（在示意图中表示为 $T_q$、$T_k$ 和 $T_v$，但实际上是一个显式卷积）。同时，softmax 函数也被弃用了。

核心思想是对长度为 $L$ 的输入序列 $u \in \mathbb{R}^{L}$ 反复应用快速评估的线性算子。
由于全局卷积参数数量庞大，其训练代价昂贵。一个显著的设计选择是使用**隐式卷积**。
与标准卷积层不同，卷积滤波器 $h$ 是通过一个小型神经网络 $\gamma_{\theta}$（也称为 Hyena Filter）隐式学习的。
该网络以位置索引和潜在的位置编码为输入。通过 $\gamma_{\theta}$ 的输出，可以构建一个 Toeplitz 矩阵 $T_h$。

这意味着，与直接学习卷积滤波器的值不同，我们学习从时间位置编码到值的映射，特别适合于长序列的计算效率。

<Tip>
值得注意的是，该映射函数可以在各种抽象模型（如神经场或状态空间模型（S4））中进行构想，如 [H3 论文](https://arxiv.org/abs/2212.14052) 中讨论的那样。
</Tip>

### 隐式卷积

线性卷积可以表示为矩阵乘法，其中一个输入被重构为一个 [Toeplitz 矩阵](https://en.wikipedia.org/wiki/Toeplitz_matrix)。

这种转化提高了参数效率。
与直接学习固定的卷积核权重值不同，这里采用了参数化函数。
该函数在网络的前向传播过程中智能地推导出核权重的值及其维度，从而优化资源使用。

<Tip>
可以通过一种直观的方式来理解隐式参数化，类似于学习一个仿射函数 $y=f(x)= a \times x + b$。与其学习每个点的位置，不如高效地学习 $a$ 和 $b$，并在需要时计算点。
</Tip>

实际中，通过 Cooley-Tukey 快速傅里叶变换（FFT）算法，加速卷积到次二次时间复杂度。也有一些加速计算的研究，如基于 Monarch 分解的 FastFFTConv。

### 汇总

![nd_hyena.png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/outlook_hyena_images/nd_hyena.png)
本质上，Hyena 可分为两步：
1. 计算一组类似于注意力的 N+1 个线性投影（可能超过 3 个投影）。
2. 投影混合：矩阵 $H(u)$ 通过矩阵乘法的组合定义。

## 为什么 Hyena 重要

H3 机制的提出接近了多头注意力机制的困惑度，但在困惑度方面仍有小差距需要弥合。

过去几年中提出了多种注意力替代方案，在探索阶段评估新架构的质量仍然具有挑战性。
创建一个能够有效处理深度神经网络中的 N 维数据的通用层，并保持良好表达能力，是一个重要的研究领域。

经验上，Hyena 算子能够在大规模下显著缩小与注意力的质量差距，以较小的计算预算达到相似的困惑度和下游性能，且无需注意力的混合。
它已经在 [DNA 序列建模](https://arxiv.org/abs/2306.15794) 中达到了最新水平，并在大型语言模型领域表现出很大潜力，如 Stripped-Hyena-7B。

类似于 Attention，Hyena 可用于计算机视觉任务。在图像分类中，Hyena 在从零开始训练 ImageNet-1k 时能够达到与注意力相当的准确性。

![hyena_vision_benchmarks.png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/outlook_hyena_images/hyena_vision_benchmarks.png)
Hyena 已应用于 N 维数据，且其 Hyena N-D 层可以作为 ViT、Swin、DeiT 骨干中的直接替换。

![vit_vs_hyenavit.png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/outlook_hyena_images/vit_vs_hyenavit.png)
图像块数量增加时在 GPU 内存效率上有显著提升。

Hyena 层级结构促进了更大、更高效的长序列卷积模型的开发。
Hyena 模型在计算机视觉领域的潜力包括：
- 处理更大、更高分辨率的图像
- 使用更小的图像块，实现细粒度的特征表示

这些特性在医疗

影像和遥感领域尤其有益。

## 向 Transformer 替代方案迈进
从简单的设计原则构建新层是一个快速发展的新兴研究领域。

H3 机制是许多基于状态空间模型（SSM）架构的基础，通常具有一种在受线性注意力启发的模块和多层感知器（MLP）模块之间交替的结构。
Hyena 作为该方法的改进，为更加高效的架构（如 Mamba 及其视觉变体）铺平了道路。

## 进一步阅读
- Hyena 官方库: [用于序列建模的卷积](https://github.com/HazyResearch/safari)
- 关于次二次模型的景观: [The Safari of Deep Signal Processing: Hyena and Beyond · Hazy Research (stanford.edu)](https://hazyresearch.stanford.edu/blog/2023-06-08-hyena-safari)
- 关于加速 FFT 算法: [FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores · Hazy Research (stanford.edu)](https://hazyresearch.stanford.edu/blog/2023-11-13-flashfftconv)
- 关于次二次模型的景观: [Zoology (Blogpost 1): Measuring and Improving Recall in Efficient Language Models · Hazy Research (stanford.edu)](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)
- Hyena 应用于计算机视觉: [[2309.13600] 多维 Hyena 用于空间归纳偏差 (arxiv.org)](https://arxiv.org/abs/2309.13600)
- 改进方法: [[2401.09417] 视觉 Mamba: 基于双向状态空间模型的高效视觉表征学习 (arxiv.org)](https://arxiv.org/abs/2401.09417)