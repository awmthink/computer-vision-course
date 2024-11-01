# 损失函数

## 介绍

在深入了解用于训练诸如 CLIP 之类模型的不同损失函数之前，有必要清晰地理解什么是对比学习。对比学习是一种无监督的深度学习方法，旨在进行表示学习。其目标是开发一种数据表示，使相似的项在表示空间中靠近，而不相似的项则明显分离。

在下图中，我们有一个示例，其中我们希望将狗的表示靠近其他狗，但同时也远离猫。

![图像信息](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/contrastive_learning.png)

## 训练目标

### 对比损失

对比损失是对比学习中最早使用的训练目标之一。它以一对相似或不相似的样本为输入，目标是将相似样本在嵌入空间中靠近，并将不相似样本推开。

从技术角度来看，假设我们有一个由多个类别的输入样本 $x_n$ 组成的列表。我们希望定义一个函数，使得同类样本的嵌入在嵌入空间中接近，不同类的样本则相距较远。将其转化为数学公式，我们得到：

$$L = \mathbb{1}[y_i = y_j]||x_i - x_j||^2 + \mathbb{1}[y_i \neq y_j]\max(0, \epsilon - ||x_i - x_j||^2)$$

简单解释如下：

- 如果样本是相似的 $y_i = y_j$，那么我们最小化项 $||x_i - x_j||^2$，它对应于它们的欧氏距离，即我们希望使它们更靠近；
- 如果样本是不相似的 $(y_i \neq y_j)$，那么我们最小化项 $\max(0, \epsilon - ||x_i - x_j||^2)$，相当于在一定限度 $\epsilon$ 内最大化它们的欧氏距离，即我们希望使它们彼此更远。

## 参考文献

- [An Introduction to Contrastive Learning](https://www.baeldung.com/cs/contrastive-learning)
- [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/)