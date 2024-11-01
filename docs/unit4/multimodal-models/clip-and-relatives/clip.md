# 对比语言-图像预训练 (CLIP)

## 简介

CLIP 是一种能够通过自然语言监督掌握视觉概念的神经网络。它通过同时训练一个文本编码器和一个图像编码器，专注于一个预训练任务，即匹配图像和对应的字幕。这种架构使 CLIP 能够无缝适应各种视觉分类基准测试。它仅需接收待识别的视觉类别名称，就能展现出类似 GPT-2 和 GPT-3 模型的“零样本”学习能力。

## 对比预训练

给定一批图像-文本对，CLIP 会计算所有可能的（图像，文本）候选对之间的稠密余弦相似度矩阵。核心思想是最大化正确配对（下图中用蓝色显示）的相似度，并最小化错误配对（图中用灰色显示）的相似度。为此，他们对这些相似度分数优化了一个对称交叉熵损失。

![CLIP 对比预训练](https://images.openai.com/blob/fbc4f633-9ad4-4dc2-bd94-0b6f1feee22f/overview-a.svg)
_图片来源：OpenAI_

简单来说，我们希望图像与其对应字幕的相似度尽可能高，而图像与其他字幕的相似度尽可能低。我们也对字幕应用此逻辑，即希望字幕与其对应图像的相似度最大化，同时与其他图像的相似度最小化。

## 文本编码器和图像编码器

CLIP 的设计包含独立的图像和文本编码器，允许用户在选择上有一定的灵活性。用户可以将标准图像编码器（如 Vision Transformer）替换为 ResNet 等其他选项，或选择不同的文本编码器，以增强适应性和实验性。当然，如果更换编码器，则需要重新训练模型，因为嵌入分布将会不同。

## 应用场景

CLIP 可用于多种应用场景。以下是一些值得注意的应用场景：

- 零样本图像分类；
- 相似性搜索；
- 扩散模型条件。

## 用法

在实际应用中，通常使用图像和预定义的类别作为输入。下面的 Python 示例展示了如何使用 transformers 库来运行 CLIP。在此示例中，我们想在 `dog` 和 `cat` 之间进行图像的零样本分类。

![猫的照片](http://images.cocodataset.org/val2017/000000039769.jpg)

```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
```

运行上述代码后，得到以下概率：

- “a photo of a cat”：99.49%
- “a photo of a dog”：0.51%

## 局限性

尽管 CLIP 在零样本分类方面表现出色，但它不太可能超越专门调优的模型。此外，其泛化能力在未遇到过的数据或实例的情况下存在一定限制。论文还指出，CLIP 的有效性和偏差受类别选择的影响，在使用 Fairface 数据集的测试中，发现性别和种族分类存在显著差异，性别准确率超过 96%，种族准确率约为 93%。

## 结论

总之，来自 OpenAI 的 CLIP 模型在多模态领域引发了革命。CLIP 的与众不同之处在于其零样本学习的能力，使其能够将图像分类到未明确训练过的类别中。这种非凡的泛化能力源于其创新的训练方法，即学习将图像与文本字幕进行匹配。

## 参考文献

- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [Lilian Weng 的 CLIP 介绍](https://lilianweng.github.io/posts/2021-05-31-contrastive/#clip)