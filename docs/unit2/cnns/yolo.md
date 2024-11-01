# YOLO

## 对目标检测的简要介绍

卷积神经网络（CNN）在解决图像分类问题上迈出了重要的一步。然而，另一项重要任务仍待解决：目标检测。目标检测不仅要求对图像中的对象进行分类，还要求准确预测对象在图像中的位置（即对象的边界框的坐标）。这正是YOLO实现突破的地方。在深入了解YOLO之前，我们先回顾一下使用CNN进行目标检测算法的发展历程。

### RCNN, Fast RCNN, Faster RCNN

#### R-CNN（基于区域的卷积神经网络）
RCNN是使用卷积神经网络进行目标检测的最简单方法之一。简单来说，其基本思想是检测一个“区域”，然后使用CNN对该区域进行分类。因此，这是一个多步骤的过程。基于这个思想，RCNN论文于2012年发表[1]。

RCNN的步骤如下：

1. 使用选择性搜索算法选择一个区域。
2. 使用基于CNN的分类器从区域中分类出对象。

对于训练，论文提出以下步骤：

1. 从目标检测数据集中制作一个检测区域的数据集。
2. 在区域数据集上微调AlexNet模型。
3. 然后在目标检测数据集上使用微调后的模型。

以下是RCNN的基本流程图：
![rcnn](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Fast%20R-CNN.png)

#### Fast RCNN

Fast RCNN改进了原始RCNN，引入了以下四项改进：

- 在单阶段训练，而不是像RCNN那样多阶段训练。使用多任务损失。
- 无需磁盘存储。
- 引入ROI池化层，仅提取感兴趣区域的特征。
- 与多步骤的RCNN/SPPnet模型不同，使用多任务损失训练一个端到端模型。

![fast_rcnn](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Fast%20R-CNN.png)

#### Faster RCNN

Faster R-CNN完全消除了对选择性搜索算法的需求！这些特性使得推理时间相比Fast R-CNN提高了90%！

- 引入了RPN（区域建议网络）。RPN是一个基于注意力的模型，使模型能够“关注”图像中包含对象的区域。
- 将RPN与Fast RCNN合并，使其成为一个端到端的目标检测模型。

![Faster RCNN](https://cdn-uploads.huggingface.co/production/uploads/6141a88b3a0ec78603c9e784/n8eDqnlEvDS5SIKGoSUpz.png)

#### 特征金字塔网络（FPN）

- 特征金字塔网络是一种用于目标检测的Inception模型。
- 它首先将图像缩小到低维嵌入。
- 然后再将其放大。
- 从每个放大的图像中尝试预测输出（在这种情况下是类别）。
- 但是在相似维度的特征之间也有跳跃连接！

请参考以下图片，这些图片摘自论文。[20]

![FPN](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/FPN.png)

![FPN2](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/FPN_2.png)

## YOLO架构

YOLO在其时代是一项突破性创新。它是一个实时的目标检测器，可以通过单一网络进行端到端训练。

### YOLO之前

在YOLO之前的检测系统包括对图像块使用图像分类器。像可变形部件模型（DPM）这样的系统使用滑动窗口方法，在整个图像的均匀间隔位置上运行分类器。

其他如RCNN的方法采用两步检测。首先检测许多可能的兴趣区域，这些区域由区域建议网络生成为边界框。然后在所有建议的区域中运行分类器以做出最终预测。还需要进行后处理，例如精炼边界框、消除重复检测，并基于场景中的其他对象重新评分边界框。

这些复杂的流程缓慢且难以优化，因为每个独立的组件都必须分别训练。

### YOLO

YOLO是一个单步检测器，在一次处理过程中同时预测边界框和对象类别。这使得系统非常快——达到每秒45帧的速度。

#### 重构目标检测

YOLO将目标检测任务重新定义为单一的回归问题，即预测边界框坐标和类别概率。

在这种设计中，我们将图像划分为$S \times S$的网格。如果对象的中心位于某个网格单元中，则该网格单元负责检测该对象。我们可以定义$B$为每个单元格中要检测的最大对象数。因此，每个网格单元预测$B$个边界框，包括每个框的置信度分数。

#### 置信度

边界框的置信度分数应反映框的预测精确程度。它应接近于真实框与预测框的IOU（交并比）。如果网格不应预测框，则该值应为零。因此，这个分数应该编码框中心位于网格中的概率以及边界框的正确性。

形式上，

$$\text{confidence} := P(\text{Object}) \times \text{IOU}_{\text{pred}}^{\text{truth}}$$

#### 坐标
边界框的坐标编码为4个数 $(x, y, w, h)$。$(x, y)$ 坐标表示相对于网格单元中心的框的中心位置。宽度和高度则相对于图像尺寸进行了归一化。

#### 类别
类别概率是一个长度为 $C$ 的向量，表示在某个单元中存在物体的条件下，各个类别的条件概率。每个网格单元只预测一个向量，即每个网格单元只会分配一个类别，因此该网格单元预测的所有 $B$ 个边界框将具有相同的类别。

形式化地表示为：
$$C_i = P(\text{class}_i \mid \text{Object})$$

在测试时，我们将条件类别概率与每个边界框的置信度相乘，得到每个边界框的类别特定置信度分数。该分数同时编码了该类别在边界框中出现的概率以及预测框与物体的匹配程度。

$$
\begin{align}
C_i \times \text{confidence} &= P(\text{class}_i \mid \text{Object}) \times P(\text{Object}) \times \text{IOU}_{\text{pred}}^{\text{truth}} \\
&=P(\text{class}_i) \times \text{IOU}_{\text{pred}}^{\text{truth}}
\end{align}
$$

总结来说，我们有一张图像，将其划分为 $S \times S$ 的网格。每个网格单元包含 $B$ 个边界框，包括5个值（置信度+4个坐标）以及一个长度为 $C$ 的向量，包含各个类别的条件概率。因此，每个网格单元是一个长度为 $B \times 5 + C$ 的向量。整个网格是 $S \times S \times (B \times 5 + C)$。

因此，如果我们有一个可以将图像转换为 $S \times S \times (B \times 5 + C)$ 特征图的可学习系统，就离任务更近一步了。

#### 网络结构
在原始的 YOLOv1 设计中，输入是一个大小为 $448 \times 448$ 的RGB图像。图像被划分为 $S \times S = 7 \times 7$ 的网格，每个网格单元负责检测 $B=2$ 个边界框和 $C=20$ 个类别。

网络结构是一个简单的卷积神经网络。输入图像经过一系列卷积层处理，然后进入全连接层。最后一层的输出被重塑为 $7 \times 7 \times (2 \times 5 + 20) = 7 \times 7 \times 30$。

YOLOv1 的设计受到 GoogLeNet 的启发，使用1x1卷积来减少特征图的深度，以降低网络的参数数量和计算量。该网络包含24个卷积层，后接2个全连接层。最后一层使用线性激活函数，其他层使用泄露整流线性激活函数：

$$\text{LeakyReLU}(x) = \begin{cases}
x & \text{if } x > 0\\
0.1x & \text{otherwise}
\end{cases}$$

下图展示了 YOLOv1 的网络结构。

![v1_arch](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/yolov1_arch.png)

#### 训练
网络在图像和真实边界框上端到端地训练。损失函数为平方误差损失之和。该损失函数设计用于惩罚网络对边界框坐标、置信度和类别概率的错误预测。我们将在下一节讨论损失函数。

YOLO 每个网格单元预测多个边界框。在训练时，我们只希望一个边界框预测器对每个物体负责。我们将预测与真实边界框 IOU（交并比）最高的预测器指定为“负责”预测某个物体。这导致了边界框预测器之间的专业化。每个预测器会在预测某些尺寸、纵横比或类别的物体方面变得更擅长，从而提高整体召回率。我们将在损失函数中为网格单元 $i$ 和边界框 $b$ 编码此信息，使用 $\mathbb{1}{ib}^{\text{obj}}$ 表示。如果不包含对象，则使用 $\mathbb{1}{ib}^{\text{noobj}}$。

##### 损失函数
现在我们有一个可以将图像转换为 $S \times S \times (B\times5 + C)$ 特征图的可学习系统，我们需要对其进行训练。

一种简单的训练此类系统的函数是使用平方误差之和。我们可以对预测值和真实值之间的平方误差求和，即对边界框坐标、置信度和类别概率进行平方误差计算。

每个网格单元 $(i)$ 的损失如下：
$$
\mathcal{L}^{i} = \mathcal{L}^{i}_{\text{coord}} + \mathcal{L}^{i}_{\text{conf}} + \mathcal{L}^{i}_{\text{class}}\\
$$

$$
\begin{align*}
\mathcal{L}^{i}_{\text{coord}} &= \sum_{b=0}^{B} \mathbb{1}_{ib}^{\text{obj}} \left[ \left( \hat{x}_{ib} - x_{ib} \right)^2 + \left( \hat{y}_{ib} - y_{ib} \right)^2 + 
\left( 
    \hat{w}_{ib} - w_{ib}
\right)^2 + 
\left( 
    \hat{h}_{ib} - h_{ib}
    \right)^2
\right]\\
\mathcal{L}^{i}_{\text{conf}} &= \sum_{b=0}^{B} (\hat{\text{conf}}_{i} - \text{conf}_{i})^2\\
\mathcal{L}^{i}_{\text{class}} &= \mathbb{1}_i^\text{obj}\sum_{c=0}^{C} (\hat{P}_{i} - P_{i})^2
\end{align*}
$$

其中
- $\mathbb{1}_{ib}^{\text{obj}}$ 表示如果 $i$ 网格单元中的 $b$ 边界框负责检测物体则为1，否则为0。
- $\mathbb{1}_i^\text{obj}$ 表示如果 $i$ 网格单元包含物体则为1，否则为0。

但该损失函数未必完全符合目标检测任务。分类和定位任务的损失简单相加，权重相同。

为了解决这个问题，YOLOv1 使用加权平方误差损失。首先，我们为定位误差分配一个独立的权重 $\lambda_{\text{coord}}$，通常设为5。

因此每个网格单元 $(i)$ 的损失如下：
$$
\mathcal{L}^{i} = \lambda_{\text{coord}}
    \mathcal{L}^{i}_{\text{coord}} + \mathcal{L}^{i}_{\text{conf}} + \mathcal{L}^{i}_{\text{class}}\\
$$
此外，许多网格单元不包含物体。置信度接近于零，因此包含物体的网格单元的梯度往往会掩盖掉这些单元的梯度。这会使网络在训练过程中不稳定。

为了解决此问题，我们也降低了不包含物体的网格单元中置信度预测的损失权重。我们为置信度损失分配一个独立的权重 $\lambda_{\text{noobj}}$，通常设为0.5。

因此每个网格单元 $(i)$ 的置信度损失如下：
$$
\mathcal{L}^{i}_{\text{conf}} = \sum_{b=0}^{B} \left[
    \mathbb{1}_{ib}^{\text{obj}} \left( \hat{\text{conf}}_{i} - \text{conf}_{i} \right)^2 +
    \lambda_{\text{noobj}} \mathbb{1}_{ib}^{\text{noobj}} \left( \hat{\text{conf}}_{i} - \text{conf}_{i} \right)^2
\right]
$$
边界框坐标的平方误差可能存在问题。它对大框和小框的误差同等对待。在大框中小的偏差不应像在小框中那样受到惩罚。

为了解决此问题，YOLOv1 对边界框宽度和高度的 **平方根** 使用平方误差损失。这使得损失函数具有尺度不变性。

因此每个网格单元 $(i)$ 的定位损失如下：
$$
\mathcal{L}^{i}_{\text{coord}} = \sum_{b=0}^{B} \mathbb{1}_{ib}^{\text{obj}} \left[ \left( \hat{x}_{ib} - x_{ib} \right)^2 + \left( \hat{y}_{ib} - y_{ib} \right)^2 +
\left(
    \sqrt{\hat{w}_{ib}} - \sqrt{w_{ib}}
\right)^2 +
\left(
    \sqrt{\hat{h}_{ib}} - \sqrt{h_{ib}}
\right)^2
\right]
$$

#### 推断 (Inference)
推断过程非常简单。我们将图像输入网络并获得\\( S \times S \times (B \times 5 + C) \\) 的特征图。接着，过滤出置信度分数低于某个阈值的框。

##### 非极大值抑制 (Non-Maximum Suppression)
在少数情况下，对于大型物体，网络可能会从多个网格单元预测多个框。为了消除重复检测，我们使用一种称为非极大值抑制 (NMS) 的技术。NMS 通过选择置信度分数最高的框，并消除所有 IOU 大于某个阈值的其他框。这个过程会迭代进行，直到没有重叠的框。

端到端流程如下所示：
![nms](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/object-detection-gif.gif)

## YOLO 的演变
到目前为止，我们已经了解了 YOLO 的基本特性以及它如何实现高度精确且快速的预测。这实际上是 YOLO 的第一个版本，称为 YOLOv1。YOLOv1 于 2015 年发布，自那以后，多个版本陆续发布。它在准确性和速度方面具有突破性，因为它引入了使用单一卷积神经网络 (CNN) 一次性处理整个图像的概念，将图像分割成 \\( S \times S \\) 的网格。每个网格单元直接预测边界框和类别概率。然而，它在小图像区域中多个物体的检测和定位精度方面表现欠佳。在接下来的几年里，不同的团队发布了许多新版本，逐步提高了精确性、速度和鲁棒性。

### **YOLOv2 (2016)**
在发布第一个版本的一年后，YOLOv2[5] 问世。该改进重点是精度和速度，并解决了定位问题。首先，YOLOv2 用 Darknet-19 取代了 YOLOv1 的主干架构，Darknet-19 是 Darknet 架构的变体。Darknet-19 比之前版本的主干更轻巧，由 19 个卷积层和最大池化层组成。这使得 YOLOv2 能够捕获更多信息。同时，它将批归一化应用于所有卷积层，因此移除了 dropout 层，解决了过拟合问题并提高了 mAP。它还引入了 anchor boxes（锚框）的概念，为检测框的宽度和高度增加了先验知识。此外，为了改善定位问题，YOLOv2 预测每个 anchor box 和网格单元（现在为 13x13）中的类别和对象。因此，对于 5 个 anchor box，最多可以生成 13x13x5 = 845 个框。

### **YOLOv3 (2018)**
YOLOv3[6] 再次显著提高了检测速度和精确性，通过用更复杂但高效的 Darknet-53 替代了 Darknet-19 架构。同时，它通过使用三种不同的尺度（13x13、26x26 和 52x52 网格）进行对象检测来更好地解决定位问题。这帮助在同一区域找到不同大小的物体。它增加了边界框数量：\\( 13 \times 13 \times 3 + 26 \times 26 \times 3 + 52 \times 52 \times 3 = 10,647 \\)。非极大值抑制 (NMS) 仍然用于筛选冗余的重叠框。

### **YOLOv4 (2020)**
在 2020 年，YOLOv4[7] 成为速度和精确性方面最佳的检测模型之一，在对象检测基准测试中达到了最新水平。作者再次更改了主干架构，选择了速度更快且更精确的 CSPDarknet53[8]。此版本的重要改进是资源利用效率的优化，使其适合在各种硬件平台上部署，包括边缘设备。此外，它在训练前增加了许多增强方法，进一步提高了模型的泛化能力。这些改进被包含在称为 bag-of-freebies 的一组方法中。Bag-of-freebies 是一些训练过程中带来成本，但在实时检测中不增加推断时间的优化方法，旨在提高模型的精度。

### **YOLOv5 (2020)**
YOLOv5[9] 将 Darknet 框架（用 C 语言编写）转换为更灵活且易于使用的 PyTorch 框架。该版本自动化了以前的 anchor 检测机制，引入了自动锚点 (auto-anchors)。自动锚点通过 k-means 和遗传算法自动训练模型的 anchor，以匹配数据。在训练期间，YOLO 自动使用这些方法优化新的更好匹配的 anchor，并将其置回 YOLO 模型。此外，它提供了不同型号的模型，依赖于硬件限制，与今天的 YOLOv8 模型命名类似：YOLOv5s、YOLOv5m、YOLOv5l 和 YOLOv5x。

### **YOLOv6 (2022)**
下一个版本 YOLOv6[10][11] 由美团视觉 AI 部门发布，文章标题为“YOLOv6：面向工业的单阶段对象检测框架”。该团队通过以下五个方面进一步提高了速度和精确性：
1) 使用 RepVGG 技术的重新参数化，它是带跳跃连接的 VGG 修改版。在推断期间，这些连接会融合以提高速度。
2) 基于重新参数化的检测器量化，添加称为 Rep-PANs 的模块。
3) 考虑到不同硬件成本和能力对模型部署的重要性。作者特别使用低功率 GPU（如 Tesla T4）进行延迟测试，而前作主要使用高成本的机器（如 V100）。
4) 引入新类型的损失函数，如用于分类的 Varifocal Loss、用于边框回归的 IoU 系列损失、用于分布的焦点损失。
5) 使用知识蒸馏技术在训练过程中提高准确性。
2023 年，YOLOv6 v3[12] 发布，标题为“YOLOv6 v3.0: A Full-Scale Reload”，进一步在网络架构和训练方案方面进行了改进，再次提升了速度和精确性（基于 COCO 数据集评估）并超越了之前发布的版本。

### **YOLOv7 (2022)**
YOLOv7 发布于文章“YOLOv7：新状态的实时对象检测器”[13][14]，由 YOLOv4 的作者发布。该版本的 bag-of-freebies 包含一种新的标签分配方法，称为粗到细引导标签分配，并使用梯度流传播路径来分析如何将重新参数化的卷积与不同网络结合。他们还提出了“扩展”和“复合缩放”方法，使实时对象检测器能够有效利用参数和计算。所有这些改进再次将实时对象检测推向了新的技术前沿，超越了之前的版本。

### **YOLOv8 (2023)**
YOLOv8[15] 由 Ultralytics 于 2023 年开发，再次成为新的技术顶尖。它在主干和颈部进行了改进，并引入了无锚框（anchor-free）方法，省去了预定义 anchor 的需求，而是直接进行预测。该版本支持广泛的视觉任务，包括分类、分割和姿态估计。此外，YOLOv8 具有可扩展性，并提供了多种预训练模型尺寸：nano、small、medium、large 和 extra-large，可轻松在自定义数据集上微调。

### **YOLOv9 (2024)**
YOLOv9 伴随题为“YOLOv9：使用可编程梯度信息实现目标学习”[16][17] 的论文发布，由 YOLOv7 和 YOLOv4 的同一作者撰写。本文着重指出了现有方法和架构在层级特征提取和空间变换过程中出现的信息损失问题。为了解决此问题，作者提出了：
* 可编程梯度信息 (PGI) 的概念，用以应对深度网络在实现多种目标时所需的不同变化。
* 通用高效层级聚合网络 (GELAN)，一种新型轻量级网络架构，相比当前方法，能够更好地利用参数，而不牺牲计算效率。

通过这些改进，YOLOv9 在 MS COCO 挑战赛上设立了新基准。

考虑到模型的时间线和不同的许可情况，我们可以绘制如下图示：
![yolo_evolution](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/yolo_evolution.png)

### **关于不同版本的说明**
本章以线性方式介绍了 YOLO 的历史/演变。然而，实际情况并非如此——许多其他 YOLO 版本是并行发布的。请注意

 YOLOv4 和 YOLOv5 在同一年发布。我们没有覆盖的其他版本包括基于 YOLOv3 (2018) 的 YOLOvX (2021) 和基于 YOLOv4 (2020) 的 YOLOR (2021) 等。
同时，理解‘最佳’模型版本的选择取决于用户的要求，如速度、精度、硬件限制和用户友好性也非常重要。例如，YOLOv2 在速度方面表现优异。YOLOv3 在准确性和速度之间提供了平衡。YOLOv4 则在适应性或跨不同硬件的兼容性方面具有最佳性能。

## 参考

[1] [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524v5) <br />
[2] [Fast R-CNN](https://arxiv.org/abs/1504.08083) <br />
[3] [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) <br />
[4] [Feature Pyramid Network](https://arxiv.org/pdf/1612.03144.pdf) <br />
[5] [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) <br />
[6] [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) <br />
[7] [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934) <br />
[8] [YOLOv4 GitHub repo](https://github.com/AlexeyAB/darknet) <br />
[9] [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) <br />
[10] [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976) <br />
[11] [YOLOv6 GitHub repo](https://github.com/meituan/YOLOv6) <br />
[12] [YOLOv6 v3.0: A Full-Scale Reloading](https://arxiv.org/abs/2301.05586) <br />
[13] [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) <br />
[14] [YOLOv7 GitHub repo](https://github.com/WongKinYiu/yolov7) <br />
[15] [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) <br />
[16] [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616) <br />
[17] [YOLOv9 GitHub repo](https://github.com/WongKinYiu/yolov9) <br />
[18] [YOLOvX](https://yolovx.com/) <br />
[19] [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)
[20] [Feature Pyramid network Paper](https://arxiv.org/abs/1612.03144)
