# 图像

在计算机视觉课程中向您解释图像是什么可能听起来有点奇怪。大概您来到这里是因为您想了解更多关于处理图像和视频格式的知识。这看似简单，但您将会大吃一惊！当谈到图像时，其中包含的内容远比眼睛所见的要多（双关意图）。

## 图像的定义

图像是对物体、场景、人物甚至概念的视觉表示。它们可以是照片、绘画、素描、图表、扫描等！更令人惊讶的是，图像也是一种函数。更准确地说，图像是一个n维函数。我们首先将其视为二维$n=2$。我们称之为$F(X,Y)$，其中$X$和$Y$是空间坐标。不要被这个复杂的名称分心。空间坐标只是我们用来描述物理空间中物体位置的系统，最常见的是二维笛卡尔坐标系。函数$F$在坐标对$x_i, y_i$处的幅度是该点的图像亮度或灰度级。亮度是让您感知光亮和黑暗的关键因素。通常，当我们拥有坐标对$x_1$和$y_1$时，我们称它们为像素（图像元素）。

图像是离散的，但组装图像的过程是连续的。图像生成过程将在下一章讨论。目前重要的是$F$在特定坐标处的值具有物理意义。函数$F(X,Y)$由两个组成部分决定：来自光源的照明量和场景中物体反射的照明量。强度图像在亮度方面也受到限制，因为该函数通常是非负的，并且其值是有限的。

这并不是唯一创建图像的方法。有时，它们是通过计算机生成的，是否借助AI并不一定。我们为由AI协助生成的图像专门设置了一章。我们将在这里介绍的大多数术语仍然适用。

另一种图像类型是体积图像或3D图像。3D图像的维度数量等于三。因此，我们有一个$F(X,Y,Z)$函数。我们大部分的推理仍然适用，唯一的不同是三元组$x_i,y_i,z_i$称为体素（体积元素）。这些图像可以通过3D方式获取，即在3D空间中重建图像。此类图像的示例包括医学扫描、磁共振成像和某些类型的显微镜成像。也可以从2D图像中重建3D图像。重建是一项具有挑战性的任务，也有专门的章节讨论它。

既然我们讨论了空间，可以来谈谈颜色了。好消息是您可能已经听说过图像通道。您可能不完全理解它们的含义，但不要担心！图像通道只是构成图像的不同颜色成分。对于$F(X,Y)$，我们将对每种颜色成分有一个$F$。每种颜色都有其独特的亮度水平。对于红色通道来说，高亮度意味着颜色非常红，而低亮度意味着几乎没有红色。

如果您只查看一种颜色的$F(x,y)$，它的范围通常是0到255，其中0表示没有亮度，255表示最大亮度。在不同的颜色系统中，这些值的组合可能有所不同。因此，在解释这些值时了解数据的来源非常重要。

还有一些特殊类型的图像，其中坐标$F(x_i,y_i)$并不描述亮度值，而是标记像素。例如将前景和背景分开的操作生成的图像就是这种类型。前景的所有内容标记为1，背景的所有内容标记为0。这些图像通常称为标签图像。当只有两个标签时，例如我们的例子，我们称之为二值图像或掩码。

您可能听说过4D或5D图像。这种术语主要用于生物医学领域和显微镜学家。再次放心！这种命名来源于那些对时间、不同通道或不同成像模式（如照片和X射线）进行体积数据成像的人。其思想是每增加一个信息源，就增加一个维度。因此，5D图像是一个在时间（4D）和使用不同通道（5D）中获取的体积图像（3D）。

那么图像在计算机中是如何表示的？最常见的是通过矩阵表示。将图像视为二维数值数组非常容易。这是一个优势，因为计算机非常擅长处理数组。将矩阵视为图像有助于理解卷积神经网络中的一些处理过程和图像预处理。稍后我们将详细讨论。

此外，图像还可以表示为图，每个节点是一个坐标，边是相邻坐标。这一点值得思考。这也意味着用于图的算法和模型也可以用于图像！反之亦然——您可以将图转换为图像并像处理图片一样分析它。

到目前为止，我们提出了一个相当灵活的图像定义。这个定义可以容纳不同的视觉数据获取方式，但它们都强调了一个关键方面：图像是包含大量空间信息的数据点。关键的不同之处在于空间分辨率（2D或3D）、颜色系统（RGB或其他）以及是否附带时间组件。

## 图像与其他数据类型的比较

### 图像与视频的区别

如果您一直在关注，您可能已经注意到视频是带有时间维度的图像的视觉表示。对于2D图像获取，您可以添加时间维度，使得$F(X,Y,T)$成为您的成像函数。

图像自然可能隐藏时间维度。毕竟，它们是在某个特定时间点拍摄的，不同的图像可能也具有时间上的联系。然而，图像和视频在如何采样这种时间信息上有所不同。图像是特定时刻的静态表示，而视频是一系列图像以一定速度播放，形成运动的幻觉。这种速度就是我们所说的帧率。

这是如此基本，以至于本课程专门设置了一章来讨论视频。在那里，我们将介绍处理该附加维度所需的适应。

### 图像与表格数据的区别

在表格数据中，维度通常由描述单个数据点的特征（列）数量定义。在视觉数据中，维度通常指描述数据的维数。对于2D图像，我们通常将数值$x_i$和$y_i$视为图像尺寸。

另一个方面是生成描述视觉数据的特征。它们可以通过传统预处理生成，也可以通过深度学习方法学习。我们称之为特征提取。它包括将在特征提取章节中详细讨论的不同算法。与表格数据的特征工程形成对比，后者是在现有特征的基础上构建新特征。

表格数据通常需要处理缺失值、编码分类变量和重新缩放数值特征。图像数据的类似过程是图像调整大小和亮度值归一化。我们称这些过程为预处理，将在“计算机视觉预处理”章节中详细讨论。

### 关键区别

下表总结了不同数据类型的关键方面。

|   | 特征                    | 图像                                                | 视频                                                 | 音频                                                                     | 表格数据                                                         |
|---|------------------------|------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------|
| 1 | 类型                   | 单一时间点                                            | 时间序列图像                                         | 单一时间点                                                              | 以行和列结构化的数据                                                |
| 2 | 数据表示              | 通常为像素的二维数组                                 | 通常为帧的三维数组                                  | 通常为音频样本的一维数组                                                 | 通常为特征列和单独数据样本行的二维数组（如电子表格、数据库表） |
| 3 | 文件类型              | JPEG、PNG、RAW等                                      | MP4、AVI、MOV等                                      | WAV、MP3、FLAC等                                                         | CSV、Excel (.xlsx、.xls)、数据库格式等                               |
| 4 | 数据增强	            | 翻转、旋转、裁剪                                      | 时间抖动、速度变化、遮挡                             | 添加背景噪音、混响、频谱操作                                            | ROSE、SMOTE、ADASYN                                                 |
| 5 | 特征提取              | 边缘、纹理、颜色                                      | 边缘、纹理、颜色、光流、轨迹                         | 频谱图、梅尔频率倒谱系数(MFCC)、Chroma特征                             | 统计分析、特征工程、数据聚合                                       |
| 6 | 学习模型               | CNNs                                                 | RNNs、3D CNNs                                        | CNNs、RNNs                                                                | 线性回归、决策树、随机森林、梯度提升                                |
| 7 | 机器学习任务          | 图像分类、分割、目标检测                              | 视频动作识别、时间建模、跟踪                           | 语音识别、说话人识别、音乐风格分类                                      | 回归、分类、聚类                                                    |
| 8 | 计算成本               | 较少                                                 | 较高                                                 | 中等到较高                                                               | 相对于其他类型通常成本较低                                          |
| 9 | 应用                   | 安全访问控制的面部识别                                | 实时通信的手语翻译                                  | 语音助手、语音转文字、音乐风格分类                                       | 预测建模、欺诈检测、天气预测                                       |