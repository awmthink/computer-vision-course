# 对扩散模型的控制

## Dreambooth

尽管扩散模型和GAN能够生成许多独特的图像，但它们并不总能精确生成所需的内容。因此，需要对模型进行微调，这通常需要大量数据和计算资源。然而，有些技术可以使用少量样本对模型进行个性化调整。

其中一个例子是Google Research的Dreambooth，这是一种训练技术，通过在少量主体或风格的图像上进行训练来更新整个扩散模型。它通过在提示词中关联一个特殊词语与示例图像来工作。有关Dreambooth的详细信息可以在[论文](https://dreambooth.github.io/)和[Hugging Face Dreambooth训练文档](https://huggingface.co/docs/diffusers/training/dreambooth)中找到。

下面可以看到Dreambooth用于训练4张狗狗图像以及一些推理示例的结果。
![Dreambooth Dog Example](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/teaser_static.jpg)
可以按照上述Hugging Face文档再现这些结果。

从这个示例可以看出，模型已经学习到该特定狗狗的特征，能够生成在不同姿势和背景下的该狗狗的新图像。尽管在计算、数据和时间方面有所改进，其他人也找到了更高效的模型定制方法。

这时，当前最流行的方法之一出现了，即低秩适应（LoRA）。此方法最初由微软在这篇[论文](https://arxiv.org/abs/2106.09685)中开发，用于高效微调大型语言模型。其主要思想是将权重更新矩阵分解为两个低秩矩阵，这些矩阵在训练过程中被优化，而模型的其余部分保持冻结状态。[Hugging Face文档](https://huggingface.co/docs/peft/conceptual_guides/lora)提供了关于LoRA如何工作的概念指南。

现在，如果我们将这些想法结合起来，就可以使用LoRA在少量样本上高效地微调扩散模型，使用Dreambooth。一个关于如何执行此操作的Google Colab教程笔记本可以在[这里](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb)找到。

由于此方法的质量和效率，许多人创建了自己的LoRA参数，许多可以在名为[Civitai](https://civitai.com/models)的网站和[Hugging Face](https://huggingface.co/collections/multimodalart/awesome-sdxl-loras-64f9af6d5cce4f4e8f351466)上找到。在Civitai上可以下载通常大小为50-500MB的LoRA权重，而在Hugging Face版本中则可以直接从模型库加载模型。
下面是如何在两种情况下加载LoRA权重并将它们与模型融合的示例。

我们可以从安装`diffusers`库开始。
```bash
pip install diffusers
````
我们将初始化`StableDiffusionXLPipeline`并加载LoRA适配器权重。
```python
from diffusers import StableDiffusionXLPipeline
import torch

model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16)
pipe.load_lora_weights(
    "lora_weights.safetensors"
)  # 如果希望从权重文件安装
pipe.load_lora_weights(
    "ostris/crayon_style_lora_sdxl"
)  # 如果希望直接从仓库安装LoRA
pipe.fuse_lora(lora_scale=0.8)
```

这使得加载一个定制的扩散模型并用于推理变得快捷，尤其是有很多模型可供选择。然后，如果我们想要移除LoRA权重，可以调用`pipe.unfuse_lora()`，这将使模型恢复到原始状态。对于`lora_scale`参数，这是一个控制推理过程中LoRA权重使用程度的超参数。值为1.0表示LoRA权重完全使用，值为0.0表示不使用LoRA权重。最佳值通常在0.7到1.0之间，但值得尝试不同的值以找到最适合的应用场景。

可以在这个Hugging Face Gradio演示中试用一些Hugging Face的LoRA模型：
<iframe
	src="https://multimodalart-LoraTheExplorer.hf.space"
	frameborder="0"
	width="850"
	height="450">
</iframe>

## 使用ControlNet引导扩散

扩散模型有多种方式可以被引导生成期望的输出，如提示词、负面提示词、引导尺度、修复绘制等。这里我们将聚焦于一种可以与所有其他方法结合并具有多种变体的方法，称为ControlNet。这一方法由斯坦福大学在这篇[论文](https://arxiv.org/abs/2302.05543)中提出。该方法允许我们通过包含深度、姿势、边缘等特定信息的图像来引导扩散模型生成更一致的图像，解决扩散模型常见的一致性问题。

ControlNet可以在文本到图像和图像到图像之间使用。下面是一个文本到图像的示例，使用了在边缘检测条件下训练的ControlNet，左上角的图像被用作输入。
可以看到，所有生成的图像形状非常相似但颜色各异。这是因为ControlNet正在引导扩散模型生成具有与输入图像相同形状的图像。

 ![bird](https://github.com/lllyasviel/ControlNet/raw/main/github_page/p1.png)

要在Stable Diffusion XL上运行ControlNet代码，请参考官方文档[这里](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetPipeline)。如果只是想测试一些示例，可以看看这个让你试验不同类型ControlNet的Gradio演示：

<iframe
	src="https://hysts-ControlNet-v1-1.hf.space"
	frameborder="0"
	width="850"
	height="450">
</iframe>