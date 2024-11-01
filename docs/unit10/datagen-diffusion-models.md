# 基于扩散模型的合成数据生成

设想要训练一个肿瘤分割模型。由于医学影像数据难以收集，模型收敛将会非常困难。理想情况下，我们希望至少有足够的数据来建立一个简单的基线模型，但如果只有少量样本呢？合成数据生成方法尝试解决这一困境，随着生成模型的兴起，我们有了更多的选择！

如前所述，可以使用生成模型（如DCGAN）来生成合成图像。本节中，我们将专注于使用[diffusers](https://huggingface.co/docs/diffusers/index)进行的扩散模型！

## 扩散模型回顾

扩散模型是一类生成模型，近年来因其生成高质量图像的能力而受到广泛关注。如今，它们被广泛应用于图像、视频和文本合成。

扩散模型通过逐步学习去噪随机高斯噪声进行工作。训练过程要求对输入样本添加高斯噪声，并让模型学习去噪。

扩散模型通常被设定为某种输入条件，除了数据分布外，还可以是文本提示、图像甚至音频。此外，还可以[构建无条件生成器](https://huggingface.co/docs/diffusers/training/unconditional_training)。

模型的内部机制有很多底层概念，但简化后的原理如下：

首先，模型向输入添加噪声并进行处理。

![noising](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/noising.jpg?download=true)

然后模型学习去噪给定的数据分布。

![denoising](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/denoising.jpg?download=true)

我们不会深入探讨理论，但理解扩散模型的工作原理会在选择生成合成数据的技术时非常有帮助。

## 文本到图像扩散模型：Stable Diffusion

本质上，Stable Diffusion（SD）的工作方式与上述描述的相同。它利用三个主要组件来生成高质量的图像。

1. **扩散过程：** 输入被多次处理以生成关于图像的有用信息。“有用性”是在训练模型时学习的。

2. **图像编码器和解码器模型：** 允许模型将图像从像素空间压缩到较小的维度空间，从而抽象出无意义的信息，提高性能。

3. **可选的条件编码器：** 此组件用于根据输入进行生成过程的条件设定。这个额外输入可以是文本提示、图像、音频和其他表示形式。最初，它是一个文本编码器。

因此，整体工作流程如下所示：
![general stable diffusion workflow](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/general-workflow.png?download=true)

我们最初使用一个文本编码器将模型设定为文本提示：
![original stable diffusion workflow](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/stable-diffusion-workflow.png?download=true)

这只是冰山一角！如果你希望深入了解Stable Diffusion（或扩散模型）的理论，可以参考[进一步阅读部分](#further-reading-about-stable-diffusion)！

[diffusers](https://huggingface.co/docs/diffusers/index)为我们提供了现成的不同任务的管道，例如：

| 任务 | 描述 | 管道 |
|------|------|------|
| 无条件图像生成 | 从高斯噪声生成图像 | [unconditional_image_generation](https://huggingface.co/docs/diffusers/using-diffusers/unconditional_image_generation) |
| 文本引导的图像生成 | 给定文本提示生成图像 | [conditional_image_generation](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation) |
| 文本引导的图像到图像翻译 | 通过文本提示调整图像 | [img2img](https://huggingface.co/docs/diffusers/using-diffusers/img2img) |
| 文本引导的图像修复 | 给定图像、掩码和文本提示，填充图像中被遮挡的部分 | [inpaint](https://huggingface.co/docs/diffusers/using-diffusers/inpaint) |
| 文本引导的深度到图像翻译 | 通过文本提示调整图像部分，同时保持结构的深度估计 | [depth2img](https://huggingface.co/docs/diffusers/using-diffusers/depth2img) |

[Diffusers Summary](https://huggingface.co/docs/diffusers/api/pipelines/overview#diffusers-summary)表格中可以找到支持任务的完整列表。

这意味着我们有很多工具可以用来生成合成数据！

## 合成数据生成方法

通常需要合成数据的三种情况：

**扩展现有数据集：**

- **样本数量不足：** 一个不错的例子是医学影像数据集，如[DDSM](https://www.mammoimage.org/databases/)（数字乳腺摄影筛查数据库，约2500个样本），少量样本使进一步分析模型的构建变得困难。建立此类医学影像数据集的成本也相当高。

**从头创建数据集：**

- **没有任何样本：** 假设你想在CCTV视频流上构建一个武器检测系统，但没有特定武器的样本，可以使用不同设置下的类似观测数据进行风格迁移，使它们看起来像CCTV流！

**保护隐私：**

- 医院收集大量患者数据，监控摄像头捕获个人面部和活动的原始信息，这些都可能侵犯隐私。我们可以使用扩散模型生成隐私保护数据集来开发我们的解决方案，而不侵犯任何人的隐私权。

我们可以通过多种方法利用文本到图像扩散模型生成定制化输出。例如，通过简单利用预训练扩散模型（如[Stable Diffusion XL](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)），你可以构建合适的提示生成图像。但生成图像的质量可能不一致，并且构建适用于特定用例的提示可能非常困难。

通常，你需要更改模型的某些部分来生成所需的个性化输出，以下是几种可以使用的技术：

**使用[Textual Inversion](https://huggingface.co/docs/diffusers/main/en/training/text_inversion)进行训练：**

![textual-inversion](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/textual_inversion.png?download=true)

文本嵌入干预（Textual Inversion）是一种通过干预模型架构中的文本嵌入来工作的技术。你可以将新标记添加到词汇表中，然后使用少量示例微调嵌入。

通过提供与新标记对应的样本，我们试图优化嵌入以捕捉对象的特征。

**训练一个[LoRA (低秩适配)](https://huggingface.co/docs/diffusers/main/en/training/lora?installation=PyTorch)模型：**

LoRA解决了微调大型语言模型的问题。它通过低秩分解将权重更新表示为两个较小的更新矩阵，从而显著减少参数数量。这些矩阵可以被训练以适应新数据！

在整个过程中，基础模型的权重保持冻结状态，因此我们只训练新的更新矩阵。最后，适应后的权重与原始权重相结合。

这意味着训练LoRA比全模型微调更快！重要的是，LoRA可以与其他技术结合使用，因为它可以直接添加到模型本身之上。

**使用[DreamBooth](https://huggingface.co/docs/diffusers/main/en/training/dreambooth)进行训练：**

![dreambooth](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/dreambooth.png?download=true)

DreamBooth是一种微调模型以个性化输出的技术。给定主体的少量图像，它允许微调预训练的文本到图像模型。主要思想是将一个唯一标识符与该特定主体关联。

训练时，我们使用词汇表中的标记并使用稀有标记标识符构建数据集。因为如果选择一个相对常见的标识符，模型还需要学习与原始含义解耦。

在原始论文中，作者在词汇表中找到稀有标记并从中选择标识符。这减少了标识符具有强先验的风险。最佳结果是在微调模型的所有层时实现的。

**使用[Custom Diffusion](https://huggingface.co/docs/diffusers/main/en/training/custom_diff

usion)进行训练：**

![custom diffusion](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-diffusion-models/custom_diffusion.png?download=true)

Custom Diffusion是一种非常强大的个性化模型技术。与前述方法相似，它仅需少量样本，但其强大之处在于能够同时学习多个概念！

它通过训练扩散过程的一部分和上述的文本编码器来工作，意味着需要优化的参数更少。因此，这种方法还实现了快速微调！

## 使用扩散模型进行数据集生成的实际案例

以下是扩散模型用于生成合成数据集的独特案例！

- [果园中苹果检测的苹果图像](https://arxiv.org/abs/2306.09762)
- [医学图像分割的自动标注息肉图像](https://arxiv.org/abs/2310.16794)
- [使用DDPMs生成3D医学图像](https://www.nature.com/articles/s41598-023-34341-2.pdf)
- [使用DDPM生成输电线路图像](https://ieeexplore.ieee.org/document/10281144)
- [无人机检测的合成航拍数据集](https://ieeexplore.ieee.org/document/10195076)
- [保护隐私的扩散模型生成合成图像](https://arxiv.org/pdf/2302.13861.pdf)

## 进一步阅读Stable Diffusion

- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
- [Diffusion Explainer: Stable Diffusion Explained with Visualization](https://poloclub.github.io/diffusion-explainer/)
- [Introduction to Diffusion Models](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
- [FastAI, Practical Deep Learning for Coders - Lesson 9: Stable Diffusion](https://course.fast.ai/Lessons/lesson9.html)
- [原始论文](https://arxiv.org/abs/2112.10752)