# 卷积神经网络简介

在上一单元中，我们学习了视觉、图像和计算机视觉的基本原理，并探讨了如何利用计算机分析图像中的视觉特征。

我们讨论的方法现在通常被称为“经典”计算机视觉。尽管这些方法在许多小规模和受限的数据集和设置中效果良好，但在处理大规模的现实世界数据集时，经典方法的局限性也逐渐显现出来。

在本单元中，我们将学习卷积神经网络，这是提升计算机视觉规模和性能的重要一步。

## 卷积：基本概念

卷积是一种用于从数据中提取特征的操作。数据可以是1D、2D或3D的。我们将通过一个具体的例子来解释这一操作。现在你需要知道的是，这一操作只是取一个由数字组成的矩阵，在数据中移动，并将数据与该矩阵的乘积之和计算出来。这个矩阵被称为卷积核或滤波器。你可能会问，“这与特征提取有什么关系，我该如何应用它？”别急！我们马上会解释。

为了帮助理解这个概念，我们来看一个例子。我们有这组1D数据，并将其可视化。可视化有助于理解卷积操作的效果。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/kernel_image.png" alt="Kernel Image">
</div>

我们有这个卷积核 [-1, 1]。我们将从最左边的元素开始，将卷积核放上去，乘以重叠的数字，并将它们相加。卷积核有中心点；这是其中的一个元素。在这里，我们选择中心为1（右边的元素）。在这里我们将左边设为假想的零，称为填充，稍后你会看到。如果不进行填充，我就必须开始将-1与最左边的元素相乘，而1将不会接触到最左边的元素，因此我们进行填充。让我们看看这会是什么样子。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/1d_conv.jpg" alt="1D Conv">
</div>

我将最左边的元素（目前是一个填充）与-1相乘，第一个元素（零）与1相乘并将它们相加，得到一个0，并记下来。现在，我们将卷积核移动一个位置并重复相同的步骤。再记下来，这种移动称为步长操作，通常是将卷积核移动一个像素。你也可以移动多个像素。当前的结果（卷积后的数据）是一个数组 [0, 0]。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/1d_conv_multiplication.png" alt="1D Conv Multiplication">
</div>

我们将重复这个过程，直到卷积核的右端触碰到每个元素，结果如下所示。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/1d_conv_result.png" alt="1D Conv Result">
</div>

注意到了什么吗？滤波器给出了数据中的变化率（即导数！）。这是我们可以从数据中提取的一个特征。让我们来可视化它。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/convolved_illustrated.png" alt="Convolved Illustrated">
</div>

卷积后的数据（卷积的结果）被称为特征图。这确实合理，因为它显示了我们可以提取的特征、与数据相关的特性以及变化率。

这正是边缘检测滤波器的工作原理！让我们看看在二维数据中会是什么样子。这次，我们的卷积核将有所不同。它将是一个3x3的卷积核（只是让你知道它也可以是2x2的）。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/2d_conv.png" alt="2D Conv">
</div>

这个滤波器非常有名，但我们暂时不会剧透给你 :)。之前的滤波器是 [-1 1]。而这个滤波器是 [-1 0 1]。它是3x3的形状，与之前没有什么不同，展示了水平轴上的增量和减量。我们来看一个例子并应用卷积。下面是我们的2D数据。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/2d_conv_matrix.png" alt="2D Conv">
</div>

把它想象成一张图像，我们希望提取水平变化。现在，滤波器的中心必须触碰到每一个像素，因此我们对图像进行填充。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/padding.png" alt="Padding">
</div>

特征图的大小将与原始数据相同。卷积的结果将写入卷积核在原始矩阵中接触的相同位置，也就是说，对于这个例子，它将接触最左边和顶部的位置。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/2d_conv_illustrated.png" alt="2D Conv">
</div>

如果我们继续应用卷积，我们会得到以下特征图。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/2d_feature_map.png" alt="2D Feature Map">
</div>

这展示了水平变化（即边缘）。这个滤波器实际上被称为Prewitt滤波器。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/prewitt_sobel.png" alt="Prewitt and Sobel">
</div>

你可以翻转[Prewitt滤波器](https://en.wikipedia.org/wiki/Prewitt_operator)来获取垂直方向的变化。[Sobel滤波器](https://en.wikipedia.org/wiki/Sobel_operator)是另一个用于边缘检测的著名滤波器。

## 卷积神经网络

那么，这与深度学习有什么关系呢？实际上，强制应用滤波器以提取特征并不适用于所有图像。设想一下，如果我们能够找到提取重要信息或检测图像中对象的最优滤波器，那将会有多好。这就是卷积神经网络的用武之地。我们使用不同的滤波器对图像进行卷积，这些特征图中的像素最终将成为我们将要优化的参数，最终我们会找到针对我们问题的最佳滤波器。

这个想法是，我们将使用滤波器来提取信息。我们随机初始化多个滤波器，创建特征图，将它们输入分类器并进行反向传播。在深入之前，我想介绍一种我们称之为“池化”的操作。

如上所见，有许多像素显示了特征图中的变化。为了知道有边缘，我们只需要看到有变化（边缘、角落、任何东西），这就足够了。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/max_pooling.png" alt="Pooling">
</div>

在上面的例子中，我们只需保留其中一个像素就足够了。通过这种方式，我们可以存储更少的参数并仍然拥有特征。这种在特征图中保留最重要元素的操作称为池化。池化使我们丢失了边缘的确切像素位置，但却能存储更少的参数。同时，这种方式使我们的特征提取机制对微小变化更具鲁棒性，例如，我们只需要知道图像中有两只眼睛、一个鼻子和一张嘴就可以判断它是一张脸，而这些元素之间的距离和大小因人而异，池化使模型在应对这些变化时更具鲁棒性。池化的另一个好处是它有助于处理不同输入大小。下面是最大池化操作，我们在每四个像素中取一个最大像素。池化有多种类型，例如平均池化、加权池化或L2池化。

让我们构建一个简单的CNN架构。我们将使用Keras示例（为了说明）并逐步解释其中的内容。下面是我们的模型（别急，我们会逐步解释发生

了什么）。

如果你不知道Keras的Sequential API在做什么，它会像乐高积木一样堆叠层并连接它们。每个层都有不同的超参数，Conv2D层接收卷积滤波器的数量、卷积核大小和激活函数，而MaxPooling2D接收池化大小，密集层接收输出单元数量（别急）。

大多数卷积神经网络实现不进行填充，以便让卷积核接触图像中的每个像素。用零填充意味着假设边界上也可能有特征，这会增加计算的复杂性。这就是为什么你会看到第一个输入大小是 (26,26)，我们在边界上丢失了一些信息。

```python
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()
```
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                16010     
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________
```

卷积神经网络从输入层和卷积层开始。Keras Conv2D层接收卷积核的数量和卷积核的大小作为参数。如下图所示。这里我们使用32个卷积核对图像进行卷积，得到32个特征图，每个特征图的大小与图像相同。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/network_illustration.png" alt="Network">
</div>

在卷积层之后，我们添加一个最大池化层以减少存储的参数数量并使模型对变化更加鲁棒，如上所述。这将减少计算的参数数量。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/network_illustration_2.png" alt="Network">
</div>

然后，这些特征图被连接并展平。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/CNNs/network_illustration_3.png" alt="Network">
</div>

接下来，我们使用一种称为dropout的操作来去除一部分参数以避免过拟合。最终，权重的最终形式将通过密集层进行分类并进行反向传播。

### 理论上的卷积神经网络中的反向传播

那么反向传播在这里是如何工作的呢？我们希望优化最优的卷积核值，所以它们是我们的权重。最终，我们希望分类器能找到像素值、卷积核和类别之间的关系。因此，我们得到一个非常长的展平数组，其中包含经过池化和激活的像素与初始权重（卷积核元素）卷积的结果。我们更新这些权重，以回答“应该使用哪些卷积核来区分猫和狗的照片？”训练卷积神经网络的重点是找到最佳的卷积核，这些卷积核是通过反向传播找到的。在卷积神经网络之前，人们会尝试在图像上应用许多滤波器来提取特征，而大多数通用滤波器（如上所示的Prewitt或Sobel）不一定适用于所有图像，因为即使在同一个数据集中，图像也可能非常不同。这也是卷积神经网络优于传统图像处理技术的原因。

使用卷积神经网络在存储方面有几个优势。

### 参数共享

在卷积神经网络中，我们在所有像素、所有通道和所有图像上使用相同的滤波器，这相比于在密集神经网络中逐像素计算参数要高效得多。这被称为“权重绑定”，这些权重被称为“绑定权重”。在自动编码器中也可以看到这种现象。

### 稀疏交互

在密集连接的神经网络中，我们一次性输入整个数据块——由于图像包含成百上千的像素，这一操作负担很重——而在卷积神经网络中，我们使用较小的卷积核来提取特征。这称为稀疏交互，它帮助我们使用更少的内存。