# 以往SOTA模型概述

## 总体趋势：

深度学习的成功，特别是基于大规模数据集（如ImageNet）训练的CNN，在图像识别领域带来了革命性变化。这一趋势在视频处理领域得以延续。然而，与静态图像相比，视频数据引入了时间维度。这一简单的变化带来了新的挑战，而基于静态图像训练的CNN并未设计用于解决这些问题。

## 视频处理领域的以往SOTA模型

### 双流网络 (2014):

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/previous sota models/SOTA Models Two-Stream architecture for video classification.png" alt="用于视频分类的双流架构">
</div>

该论文将深度卷积网络（ConvNets）扩展到视频数据中的动作识别任务中。

所提出的架构被称为双流网络。它在神经网络中使用了两个独立的路径：

- **空间流 (Spatial Stream)：** 一个标准的2D CNN处理单帧以捕获外观信息。
- **时间流 (Temporal Stream)：** 一个2D CNN或其他网络，处理多帧序列（光流）以捕获运动信息。
- **融合 (Fusion)：** 将两个流的输出结合，以便在动作识别等任务中利用外观和运动线索。

### 3D ResNets (2017):

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/previous sota models/SOTA Models Residual block. Shortcut connections bypass a signal from the top of the block to the tail. Signals are summed at the tail..png" alt="残差块。快捷连接绕过信号从块顶端到尾部，并在尾部汇总信号。">
</div>

标准3D CNN通过3D卷积核（2D空间信息 + 时间信息）同时捕获空间和时间信息。然而，这种模型的缺点在于大量参数导致训练计算开销更大，速度较2D版本慢。因此，3D版的ConvNets通常层数较少，较2D CNN的深层架构更为精简。

在此论文中，作者将ResNet架构应用于3D CNN。这一方法为3D CNN引入了更深的模型，提升了准确性。

实验表明，3D ResNets（特别是更深的ResNet-34）在较大数据集上优于C3D模型。诸如Sports-1M C3D等预训练模型在较小数据集上有助于减少过拟合。总体而言，3D ResNets能有效利用深层架构来捕获视频数据中的复杂时空模式。

| 方法                    | 验证集         |             |            | 测试集       |             |            |
|-------------------------|----------------|-------------|------------|-------------|-------------|------------|
|                         | Top-1          | Top-5       | 平均       | Top-1       | Top-5       | 平均       |
| 3D ResNet-34            | 58.0           | 81.3        | **69.7**   | -           | -           | **68.9**   |
| C3D*                    | 55.6           | 79.1        | 67.4       | 56.1        | 79.5        | 67.8       |
| C3D w/ BN               | 56.1           | 79.5        | 67.8       | -           | -           | -          |
| RGB-I3D w/o ImageNet    | -              | -           | 68.4       | 88.0        | **78.2**    |

### (2+1)D ResNets (2017):

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/previous sota models/SOTA Models 3D vs (2+1)D convolution..png" alt="3D vs (2+1)D卷积。">
</div>

(2+1)D ResNets受3D ResNets启发，但其层结构有所不同。该架构结合了2D卷积和1D卷积：

- 2D卷积捕获单帧内的空间特征。
- 1D卷积捕获连续帧间的运动信息。

该模型可以直接从视频数据中学习时空特征，有望在动作识别等视频分析任务中表现更佳。

- 优势：
    - 在两次操作间加入非线性整流(ReLU)增加了非线性数量，与相同参数数量的全3D卷积网络相比，能够表示更复杂的函数。
    - 分解简化了优化过程，在实践中可获得更低的训练损失和测试损失。

| 方法                            | Clip@1 准确率 | Video@1 准确率 | Video@5 准确率 |
|---------------------------------|--------|---------|---------|
| DeepVideo                       | 41.9   | 60.9    | 80.2    |
| C3D                             | 46.1   | 61.1    | 85.2    |
| 2D ResNet-152                   | 46.5   | 64.6    | 86.4    |
| Conv pooling                    | -      | 71.7    | 90.4    |
| P3D                             | 47.9   | 66.4    | 87.4    |
| R3D-RGB-8frame                  | 53.8   | -       | -       |
| R(2+1)D-RGB-8frame              | 56.1   | 72.0    | 91.2    |
| R(2+1)D-Flow-8frame             | 44.5   | 65.5    | 87.2    |
| R(2+1)D-Two-Stream-8frame       | -      | 72.2    | 91.4    |
| R(2+1)D-RGB-32frame             | **57.0**   | **73.0**    | **91.5**    |
| R(2+1)D-Flow-32frame            | 46.4   | 68.4    | 88.7    |
| R(2+1)D-Two-Stream-32frame      | -      | **73.3**    | **91.9**    |

## 当前研究

目前，研究人员正探索更深层的3D CNN架构。另一种前景良好的方法是将3D CNN与注意力机制等其他技术结合。此外，开发更大规模的视频数据集也成为热点，如[Kinetics](https://github.com/google-deepmind/kinetics-i3d)。

Kinetics数据集是一个大规模高质量视频数据集，常用于人类动作识别研究。该数据集包含数十万视频片段，涵盖广泛的日常人类活动。

<Tip>


</Tip>

## 结论

视频分析模型的发展令人瞩目。这些模型受到其他SOTA模型的深刻影响。例如，双流网络的灵感来源于ConvNets，而(2+1)D ResNets则借鉴了3D ResNets。随着研究的推进，预计未来会有更先进的架构和技术涌现。