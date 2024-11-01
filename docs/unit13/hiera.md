# Hiera

## 什么是 Hiera？

[Hiera](https://arxiv.org/abs/2306.00989)（分层视觉 Transformer）是一种无需其他视觉模型中专用组件即可实现高精度的架构。作者提出通过一个强大的视觉预训练任务来训练 Hiera，以去除不必要的复杂性，从而创建一个更快、更精确的模型。

![Hiera 架构](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/hiera_architecture.png)

## 从 CNN 到 ViTs

CNN 和分层模型非常适合计算机视觉任务，因为它们可以有效地捕获视觉数据的分层和空间结构。这些模型在早期阶段使用较少的通道但较高的空间分辨率来提取简单特征，而在后期阶段使用更多的通道但较低的空间分辨率来提取更复杂的特征。

![CNNs](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/CNN_architecture.webp)

另一方面，视觉 Transformer（ViTs）是更精确、可扩展且结构简单的模型，在推出后迅速席卷了计算机视觉领域。然而，这种简化也带来了缺点：它们缺乏“视觉归纳偏置”（即它们的架构并非专门为视觉数据设计）。

为弥补这一缺陷，人们尝试在 ViTs 中添加分层组件。然而，所有这些改进后的模型最终都变得更慢、更庞大且更难扩展。

## Hiera 的方法：只需预训练任务

Hiera 论文的作者认为，通过使用一种称为 MAE 的强视觉预训练任务，ViT 模型可以学习空间推理并在计算机视觉任务中表现出色，因此可以去除多阶段视觉 Transformer 中不必要的组件和复杂性，从而实现更高的精度和速度。

论文的作者实际上去除了哪些组件？为了理解这一点，我们首先介绍 [MViTv2](https://arxiv.org/abs/2112.01526)，这是 Hiera 派生的基本分层架构。MViTv2 在其四个阶段中学习多尺度表示：它从低通道容量但高空间分辨率的低级特征建模开始，然后在每个阶段交换通道容量和空间分辨率，以便在更深的层中建模更复杂的高级特征。

![MViTv2](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/mvitv2.png)

我们不会深入挖掘 MViTv2 的关键特性（因为这不是我们的主要讨论内容），而是将在下一节中简要说明这些特性，以展示研究人员如何通过简化该基本架构来创建 Hiera。

## 简化 MViTv2

![简化 MViTv2](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/hiera_changes.png)

此表列出了作者对 MViTv2 进行的所有更改，以创建 Hiera，并展示了每个更改对图像和视频的精度和速度的影响。

- **将相对位置嵌入替换为绝对位置嵌入**：MViTv2 将来自原始 [ViT](https://arxiv.org/abs/2010.11929) 论文的绝对位置嵌入更换为在每个块的注意力中添加的相对位置嵌入。作者撤销了此更改，因为它增加了模型的复杂性，并且从表中可以看出，在使用 MAE 进行训练时不需要这些相对位置嵌入（此更改后精度和速度均有所提高）。
- **去除卷积层**：由于论文的关键思想是通过一个强大的视觉预训练任务，模型可以学习空间偏置，因此去除卷积层这一特定于视觉的模块并减少潜在的额外负担成为一个重要的改变。作者首先用最大池化层替换每个卷积层，虽然这最初导致精度下降，但主要是由于它对图像特征产生了巨大影响。然而，作者意识到可以去除一些额外的最大池化层，特别是步幅为 1 的池化层，因为它们实际上只是对每个特征图应用了 ReLU。这样一来，作者几乎恢复了之前的精度，同时使图像速度提高了 22%，视频速度提高了 27%。

## Masked Autoencoder

Masked Autoencoder（MAE）是一种无监督训练范式。与其他自动编码器一样，它包括将高维数据（图像）编码为低维表示（嵌入），以便可以将这些数据解码回原始的高维数据。然而，视觉 MAE 技术包括丢弃一定数量的图像块（约 75%），编码剩余的块，然后尝试预测丢失的部分。近年来，这一理念被广泛应用于图像编码器的预训练任务。

![MAE](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/mae.png)

## Hiera 的重要性

在 Transformer 模型主导的时代，仍然有许多尝试通过增加 CNN 的复杂性将其转换为分层模型。尽管分层模型在计算机视觉中表现出色，但该研究表明，实现分层 Transformer 并不需要复杂的架构修改。相反，仅专注于训练任务本身就可以实现简单、快速且精确的模型。