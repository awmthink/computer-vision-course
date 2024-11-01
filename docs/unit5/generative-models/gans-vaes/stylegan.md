# StyleGAN 变体

本章您将学习以下内容：

- Vanilla GAN 的不足之处
- StyleGAN1 的组成部分和优点
- StyleGAN1 的缺点及引入 StyleGAN2 的必要性
- StyleGAN2 的缺点及引入 StyleGAN3 的必要性
- StyleGAN 的应用场景

## Vanilla GAN 的不足之处
生成对抗网络（GANs）是一类生成模型，用于生成逼真的图像。但是，很明显您无法控制图像的生成方式。在 Vanilla GAN 中，您拥有两个网络：(i) 生成器和 (ii) 判别器。判别器将图像作为输入并返回该图像是真实图像还是由生成器生成的合成图像。生成器接受噪声向量（通常从多变量高斯分布中采样）并尝试生成与训练样本中图像相似但不完全相同的图像，最初会生成一张杂乱的图像，但生成器的目标是逐渐欺骗判别器，让其相信生成的图像是真实的。

考虑一个训练过的 GAN，假设 z1 和 z2 是从高斯分布中采样的两个噪声向量，并输入到生成器以生成图像。假设 z1 生成的图像是一位戴眼镜的男性，而 z2 生成的图像是一位不戴眼镜的女性。如果您需要一张戴眼镜的女性的图像，这种显式的控制在 Vanilla GAN 中无法直观地实现，因为特征是纠缠的（详细信息见下文）。这一点需要一定的理解，您会在了解 StyleGAN 的成就时更深入地理解。

简而言之，StyleGAN 是对生成器结构的一种特殊修改，而判别器保持不变。StyleGAN 的这一修改使生成器能够根据用户需求生成图像，并提供对高级（如姿势、面部表情）和随机（如皮肤毛孔、局部头发位置等低级特征）的控制。除去灵活的图像生成功能，近年来，StyleGAN 还被用于诸如隐私保护、图像编辑等下游任务。

## StyleGAN1 的组成部分和优点

![Architecture](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/stylegan_arch.png)

让我们深入了解 StyleGAN 中引入的特殊组件，这些组件赋予了 StyleGAN 上述的强大功能。请不要被上图吓到，这是一个简单而强大的理念，您可以轻松理解。

正如我之前所说，StyleGAN 仅修改生成器，判别器保持不变，因此未在图中显示。图（a）对应于 ProgessiveGAN 的结构。ProgessiveGAN 只是一个 Vanilla GAN，但它不是生成固定分辨率的图像，而是逐步生成更高分辨率的图像，以期生成真实的高分辨率图像，即生成器的 block 1 生成分辨率为 4x4 的图像，block 2 生成分辨率为 8x8 的图像，以此类推。
图（b）是提出的 StyleGAN 架构。它包含以下主要组件：
1. 一个映射网络
2. 自适应实例归一化 (AdaIN)
3. 噪声向量的拼接

让我们逐一分解。

### 映射网络
在传统 GAN 中，潜在代码（也称为噪声向量）z 被直接传递给生成器，而在 StyleGAN 中，z 被映射到 w，并通过 8 层 MLP 获得。生成的潜在代码 w 不仅作为输入传递给生成器的第一层（如在 ProgessiveGAN 中那样），而是传递到生成器网络的每个块（在 StyleGAN 术语中称为合成网络）。这里有两个主要思想：

- 将潜在代码从 z 映射到 w，使特征空间解耦。这里的解耦是指在一个维度为 512 的潜在代码中，如果只改变一个特征值（例如在 512 个值中，只增加或减少第 4 个值），那么理想情况下，在解耦的特征空间中，只有一个现实世界的特征会发生变化。如果第 4 个特征值对应于现实世界特征“微笑”，那么改变 512 维潜在代码的第 4 个值应生成微笑/不微笑/中间状态的图像。
- 将潜在代码传递到每一层对真实特征的控制有显著影响。例如，将潜在代码 w 传递到合成网络的低层块，可以控制高层次的方面，如姿势、总体发型、面部形状和眼镜，而将潜在代码 w 传递到合成网络的高分辨率块则可以控制更小尺度的面部特征、发型、眼睛开闭等。

### 自适应实例归一化 (AdaIN)

![Adaptive instance normalisation](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/AdaIN.png)

AdaIN 通过允许基于来自独立源的样式信息动态调整归一化参数（均值和标准差）来修改实例归一化。这种样式信息通常来自潜在代码 w。

在 StyleGAN 中，潜在代码不是直接传递给合成网络，而是将仿射变换 w，即 y 传递给不同的块。y 被称为“样式”表示。
这里，$y_{s,i}$ 和 $y_{b,i}$ 是样式表示 y 的均值和标准差，而 $\\mu(x_i)$ 和 $\\sigma(x_i)$ 是特征图 x 的均值和标准差。

AdaIN 使生成器能够在生成过程中动态调节其行为。这在生成的输出的不同部分可能需要不同的样式或特征的情况下尤为有用。

### 噪声向量的拼接

在传统 GAN 中，生成器必须自己学习随机特征。这里的随机特征是指像头发位置、皮肤毛孔等微小但重要的细节，它们应当在不同图像生成之间变化，而不应保持不变。传统 GAN 缺乏显式结构，这使得生成器很难独立引入这些像素级的随机变化，因此通常无法产生多样化的随机特征集。

而在 StyleGAN 中，作者提出通过将噪声图添加到合成网络（即生成器）的每一块的特征图中，让每一层利用此信息生成多样的随机特性，而不像传统 GAN 那样需要独自完成。这一策略取得了成功。

![Example for noise](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/noise.png)

## StyleGAN1 的缺点及引入 StyleGAN2 的必要性
StyleGAN 在数据驱动的无条件生成图像建模中取得了业界领先的成果。然而，现有架构设计中存在一些问题，这些问题在下一个版本 StyleGAN2 中得到解决。

为了提高章节的可读性，我们不会深入探讨架构的细节，而是简单说明第一版中的特征性伪影以及如何改进质量。

本文讨论了两个主要伪影，第一个是常见的斑块状伪影，另一个是由于现有的渐进式生成架构产生的空间位置偏好伪影。

![blob Artifact](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/norm.png)

上图显示了斑块状结构，作者认为该结构源于 StyleGAN1 的归一化过程。下图（d）展示了克服此问题的改进架构。

![Demodulation](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/stylegan2_demod.png)

(ii) 修复了渐进式 GAN 结构中的强位置偏好伪影。

![Phase Artifact](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/progress.png)

在上图中，每个图像都是通过内插潜在代码 w 来调节姿势生成的。这导致了相当不真实的图像，即使视觉质量很高。

为了解决这个问题，StyleGAN2 引入了跳跃生成器和残差判别器，而不再需要渐进式生成。

StyleGAN2 还引入了其他一些变化，但上述两点是首先需要了解的重要内容。

## StyleGAN2 的缺点及引入 StyleGAN3 的必要性
StyleGAN2 的作者发现合成网络对绝对像素坐标的依赖有不良影响，这导致了别名效应现象。

![Animation of aliasing](https://huggingface.co/datasets/hwaseem04/Documentation-files/resolve/main/CV-Course/MP4%20to%20GIF%20conversion.gif)
在上图中，动画是通过内插潜

在代码 w 生成的。您可以清晰地看到，在左图中纹理像素似乎固定在位置上，只有高级特征（如面部姿势/表情）发生变化。生成这样的动画时会显得很不真实。StyleGAN3 从根本上解决了这个问题，您可以在右侧的动画中看到效果。

## 应用场景
StyleGAN 生成逼真图像的能力为多种应用打开了大门，包括图像编辑、隐私保护，甚至是创意探索。

**图像编辑**

- 图像修补：无缝而逼真地填充缺失的图像区域。
- 图像风格转换：将一种图像的风格转换到另一种图像。

**隐私保护应用**

- 生成合成数据：用逼真的合成数据替换敏感信息，以便用于训练和测试。
- 匿名化图像：模糊或更改图像中的可识别特征，以保护个人隐私。

**创意探索**

- 生成时尚设计：StyleGAN 可用于生成逼真且多样的时尚设计。
- 创建沉浸式体验：StyleGAN 可用于为游戏、教育等应用创建逼真的虚拟环境。例如，Stylenerf：一种基于样式的 3D 感知生成器，用于高分辨率图像合成。

这些只是一个不完全的列表。

## 参考文献
- StyleGAN - [repository](https://github.com/NVlabs/stylegan), [Paper](https://arxiv.org/abs/1812.04948)
- StyleGAN2 - [repository](https://github.com/NVlabs/stylegan2), [Paper](http://arxiv.org/abs/1912.04958)
- StyleGAN3 - [repository](https://github.com/NVlabs/stylegan3), [Paper](https://arxiv.org/abs/2106.12423)ssss