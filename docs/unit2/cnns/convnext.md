# ConvNext - 面向2020年代的卷积神经网络 (2022)

## 引言
近年来，视觉Transformer（ViTs）的突破迅速取代了纯CNN模型，成为图像识别的新一代技术。
有趣的是，研究发现CNN可以借鉴视觉Transformer中的许多设计选择。
ConvNext通过借鉴视觉Transformer的技术，为纯卷积模型带来了显著改进，在准确性和可扩展性方面达到了与视觉Transformer相媲美的效果。

## 关键改进
ConvNeXT论文的作者以常规的ResNet（ResNet-50）为基础，逐步现代化并改进架构，以模仿视觉Transformer的分层结构。
关键改进包括：
- 训练技术
- 宏观设计
- ResNeXt化
- 倒置瓶颈
- 大内核尺寸
- 微观设计

我们将逐一讨论这些关键改进。这些设计本身并非新颖，但可以学到研究人员如何系统地调整和修改设计来改进现有模型。
为了展示每个改进的效果，我们将在ImageNet-1K上比较模型在修改前后的准确率。

![Block Comparison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/block_comparison.png)

## 训练技术
研究人员首先意识到，虽然架构设计选择至关重要，但训练过程的质量也在很大程度上影响了性能表现。
受DeiT和Swin Transformers启发，ConvNext紧密适应了它们的训练技术。主要变化包括：
- Epochs: 将原始90个epochs扩展到300个epochs。
- Optimizer: 使用AdamW优化器替代Adam优化器，AdamW在处理权重衰减方面有所不同。
- Mixup（生成随机图像对的加权组合）、Cutmix（将图像的一部分切割并用另一图像的补丁替代）、RandAugment（应用一系列随机增强如旋转、平移和剪切）和Random Erasing（随机选择图像中的矩形区域并用随机值擦除像素）以增加训练数据。
- 正则化：使用随机深度和标签平滑作为正则化技术。

修改这些训练过程后，ResNet-50的准确率从76.1%提升至78.8%。

## 宏观设计
宏观设计指的是系统或模型中的高级结构决策和考虑事项，例如层的排列、不同阶段的计算负载分配以及整体结构。
通过检查Swin Transformer的宏观网络结构，作者发现了两个对ConvNext性能有益的设计考虑。

### 阶段计算比
阶段计算比指的是神经网络模型中各个阶段的计算负载分配。
ResNet-50有四个主要阶段，包含（3, 4, 6, 3）个块，这意味着其计算比为3:4:6:3。
为了跟随Swin Transformer的计算比1:1:3:1，研究人员将ResNet的阶段块数从（3, 4, 6, 3）调整为（3, 3, 9, 3）。
改变阶段计算比后，模型准确率从78.8%提高到79.4%。

### 更改stem为Patchify
通常在ResNet架构的开始，输入会先经过一个带有步长2的7×7卷积层stem，再通过最大池化，图像缩小4倍。
然而，作者发现，用一个4×4内核大小、步长为4的卷积层替代stem更有效，能够将图像以非重叠的4x4块进行卷积。
Patchify实现了与stem相同的图像缩小效果，但减少了层的数量。
此Patchifying步骤将模型准确率从79.4%稍微提升到79.5%。

## ResNeXt化
ConvNext也采用了ResNeXt的理念，这在前面章节中已说明。
ResNeXt在浮点运算次数（FLOPs）和准确率之间实现了优于标准ResNet的平衡。
通过使用深度卷积和1 × 1卷积，可以实现空间和通道混合的分离——这是视觉Transformer的一个特点。
使用深度卷积减少了FLOPs和准确率。
然而，通过将通道从64增加到96，保持了与原始ResNet-50相似的FLOPs数量，同时提高了准确率。
此改进使模型准确率从79.5%提升至80.5%。

## 倒置瓶颈
每个Transformer块中的一个常见理念是使用倒置瓶颈，即隐藏层远大于输入维度。
这一理念也被MobileNetV2在计算机视觉中推广。
ConvNext采用了这一思路，输入层为96个通道，将隐藏层增加到384个通道。
通过此技术，模型准确率从80.5%提升到80.6%。

![Inverted Bottleneck Comparison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/inverted_bottleneck.png)

## 大内核尺寸
视觉Transformer表现优异的关键因素之一是其非局部自注意力，允许更宽广的图像特征感受野。
在Swin Transformer中，注意力块的窗口大小至少设置为7×7，超越了ResNeXt的3x3内核大小。
然而，在调整内核大小之前，需要重新定位深度卷积层，如下图所示。
这种重新定位使1x1层可以有效地处理计算任务，而深度卷积层作为更非局部的接收器。
通过此方式，网络可以利用较大内核尺寸卷积的优势。
实现7x7内核大小保持了准确率在80.6%不变，但降低了模型整体FLOPs效率。

![Moving up the Depth Conv Layer](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/depthwise_moveup.png)

## 微观设计
除了上述的修改外，作者还对模型进行了一些微观设计的更改。
微观设计指的是低层级的结构决策，例如激活函数的选择和层细节。
一些显著的微观更改包括：
- 激活：将ReLU激活替换为GELU（高斯误差线性单元），并在残差块中移除所有GELU层，仅保留一个位于两个1×1层之间的GELU层。
- 正则化：减少正则化层，移除两个BatchNorm层并用LayerNorm替换BatchNorm，仅在conv 1 × 1层之前保留一个LayerNorm层。
- 下采样层：在ResNet阶段之间添加一个独立的下采样层。

这些最终的修改将ConvNext的准确率从80.6%提升至82.0%。
最终的ConvNext模型超过了Swin Transformer的81.3%的准确率。

## 模型代码
你可以访问[此HuggingFace文档](https://huggingface.co/docs/transformers/model_doc/convnext)来学习如何将ConvNext模型集成到代码中。

## 参考文献
论文《A ConvNet for the 2020s》由Facebook AI Research的研究团队在2022年提出，团队成员包括Zhuang Liu、Hanzi Mao、Chao-Yuan Wu、Christoph Feichtenhofer、Trevor Darrell和Saining Xie。
该论文可以在[此处](https://arxiv.org/abs/2201.03545)找到，GitHub仓库可以在[此处](https://github.com/facebookresearch/ConvNeXt)找到。