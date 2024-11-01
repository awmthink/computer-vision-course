# GoogLeNet

在本章中，我们将探讨一种名为GoogLeNet的卷积神经网络结构。

## 概述

Inception架构是一种专为图像分类和检测等计算机视觉任务设计的卷积神经网络（CNN），以其高效性脱颖而出。该架构包含不到700万参数，与前代相比显著紧凑，比AlexNet小9倍，比VGG16小22倍。该架构在ImageNet 2014竞赛中表现卓越，谷歌的GoogLeNet（向LeNet致敬的命名）在使用更少参数的情况下，设立了新的性能标杆。

### 架构创新

在Inception架构诞生之前，像AlexNet和VGG这样的模型已经展示了更深层网络结构的优势。然而，较深的网络通常带来更多的计算步骤，并可能导致过拟合和梯度消失问题。Inception架构提供了一种解决方案，使得可以在减少浮点参数数量的情况下训练复杂的CNN。

#### Inception “网络中的网络” 模块

在AlexNet或VGG等先前的网络中，基本模块就是卷积层本身。然而，Lin等人于2013年提出“网络中的网络”概念，认为单一的卷积并不一定是合适的基础模块，它应当更复杂。受到这一思想的启发，Inception模型的作者决定采用更复杂的构建模块，称为Inception模块，取名于著名电影《盗梦空间》（梦中梦）。

Inception模块主张应用不同核大小的卷积滤波器，以在多尺度下提取特征。对于任意输入特征图，Inception模块并行应用\\(1 \times 1\\)卷积、3x3卷积和5x5卷积。除此之外，还进行一次最大池化操作。所有四个操作都设置了填充和步幅，使空间维度保持一致。这些特征随后被连接并成为下一阶段的输入。参见图1。

![inception_naive](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/inception_naive.png)

图1：基础Inception模块

如图所示，使用较大卷积核（如5x5）在多尺度上进行多次卷积会显著增加参数数量。这一问题在输入特征尺寸（通道数）增加时尤为明显。随着网络层数增加并堆叠多个Inception模块，计算量将急剧增加。简单的解决方案是，在计算需求增加的地方减少特征数量。卷积层是计算需求的主要痛点。在3x3和5x5卷积之前，通过计算开销较小的\\(1 \times 1\\)卷积减少特征维度。以下示例演示了这一点。

我们想将一个\\( S \times S \times 128 \\)的特征图通过5x5卷积转为\\( S \times S \times 256 \\)。参数数量（不包括偏差）为5\*5\*128\*256 = 819,200。但若先通过\\(1 \times 1\\)卷积将特征维度减少到64，则参数数量（不包括偏差）为\\( 1\times 1\times 128\times 64 + 5\times 5\times 64\times 256 = 8,192 + 409,600 = 417,792 \\)。这样参数数量减少了将近一半！

我们还希望在将最大池化的输出特征图与其他输出特征图连接之前减少其特征数量。因此，在最大池化层之后再添加一个\\( 1\times 1 \\)卷积。每个\\( 1\times 1 \\)卷积后添加ReLU激活，增加了模块的非线性和复杂性。参见图2。

![inception_reduced](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/inception_reduced.png)

图2：改进版Inception模块

此外，由于并行进行多尺度的卷积操作，无需加深网络便能实现更多操作，从而在一定程度上缓解梯度消失问题。

#### 平均池化

在AlexNet或VGG等先前网络中，最后几层通常是全连接层。这些全连接层由于其大量单元，贡献了网络中大部分参数。例如，VGG16中89%的参数集中在最后三层全连接层中。AlexNet中95%的参数集中在最终的全连接层中。这种需求可以归因于卷积层的复杂性不足。

然而，有了Inception块后，无需全连接层，只需在空间维度上进行简单的平均池化即可。这也是从“网络中的网络”论文中得出的。然而，GoogLeNet包含了一个全连接层，据报告Top-1准确率增加了0.6%。

GoogLeNet仅有15%的参数在全连接层中。

#### 辅助分类器

通过引入计算量较小的\\( 1 \times 1 \\)卷积并用平均池化替代多个全连接层，网络参数显著减少，允许我们添加更多层并加深网络。然而，堆叠层会导致梯度消失问题，梯度在向网络的初始层传播时逐渐变小接近零。

论文提出了辅助分类器——在中间层分支出一些小分类器，并将这些分类器的损失（赋予较小权重）加到总损失中。这确保了靠近输入层的层也能接收到一定大小的梯度。

辅助分类器由以下组成：
- 具有\\( 5 \times 5 \\)滤波器大小和步幅3的平均池化层；
- 128个滤波器的\\( 1 \times 1 \\)卷积，用于降维和ReLU激活；
- 1024个单元的全连接层，ReLU激活；
- 70%的dropout层；
- Softmax损失的线性分类层。

这些辅助分类器在推理时被移除。辅助分类器带来的提升较小（0.5%）。

![googlenet_aux_clf](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/googlenet_auxiliary_classifier.jpg)

图3：辅助分类器

### GoogLeNet架构

完整的GoogLeNet架构如图所示。所有卷积操作（包括Inception块内的卷积）均使用ReLU激活。架构以两个卷积和最大池化块开始，接着是一个包含两个Inception模块（3a和3b）和一个最大池化的块。然后是一个包含5个Inception模块（4a, 4b, 4c, 4d, 4e）和一个最大池化的块。辅助分类器从4a和4d的输出中提取。最后两个Inception模块（5a和5b）后接一个平均池化层和一个128单元的全连接层。

![googlenet_arch](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/googlenet_architecture.png)
图4：完整的GoogLeNet架构

### 代码

```python
import torch
import torch.nn as nn


class BaseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BaseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_proj):
        super(InceptionModule, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.ReLU(True),
        )

        self.b2 = nn.Sequential(
            BaseConv2d(in_channels, n3x3red, kernel_size=1),
            BaseConv2d(n3x3red, n3x3, kernel_size=3, padding=1),
        )

        self.b3 = nn.Sequential(
            BaseConv2d(in_channels, n5x5red, kernel_size=1),
            BaseConv2d(n5x5red, n5x5, kernel_size=5, padding=2),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BaseConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)




class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.7):
        super(AuxiliaryClassifier, self).__init__()
        self.pool = nn.AvgPool2d(5, stride=3)
        self.conv = BaseConv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU(True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, use_aux=True):
        super(GoogLeNet, self).__init__()

        self.use_aux = use_aux
        ## block 1
        self.conv1 = BaseConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        ## block 2
        self.conv2 = BaseConv2d(64, 64, kernel_size=1)
        self.conv3 = BaseConv2d(64, 192, kernel_size=3, padding=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        ## block 3
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        ## block 4
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        ## block 5
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        ## auxiliary classifier
        if self.use_aux:
            self.aux1 = AuxiliaryClassifier(512, 1000)
            self.aux2 = AuxiliaryClassifier(528, 1000)

        ## block 6
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        ## block 1
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.lrn1(x)

        ## block 2
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)

        ## block 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        ## block 4
        x = self.inception4a(x)
        if self.use_aux:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.use_aux:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        ## block 5
        x = self.inception5a(x)
        x = self.inception5b(x)

        ## block 6
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.use_aux:
            return x, aux1, aux2
        else:
            return x