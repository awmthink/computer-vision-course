# 图像分割

图像分割是将图像划分为有意义的部分，其核心在于创建掩膜，以突出显示图像中的每个对象。这个任务的直观理解是*可以将其视为对图像中每个像素的分类*。分割模型是各行各业中的核心模型，广泛应用于农业和自动驾驶领域。在农业中，这些模型用于识别不同的土地区域和评估作物的生长阶段。在自动驾驶汽车中，它们用于识别车道、人行道和其他道路使用者。

![图像分割](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/segmentation-example.png)

根据不同的上下文和预期目标，可以应用不同类型的分割。以下是最常见的分割类型。
- **语义分割**：每个像素被赋予最有可能的类别。例如，在语义分割中，模型不区分两只独立的猫，而是关注像素的类别。这完全是关于每个像素的分类。
- **实例分割**：这种类型包括使用唯一的掩膜识别每个对象的实例。它结合了目标检测和分割的方面，用于区分同一类别的各个独立对象。
- **全景分割**：一种混合方法，结合了语义和实例分割的元素。它为每个像素分配一个类别和实例，有效地整合了图像中的*是什么*和*在哪里*的内容。

![分割类型比较](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/segmentation-types.png)

选择合适的分割类型取决于上下文和预期目标。一个很酷的事情是，最近的模型允许您使用单一模型实现这三种分割类型。我们推荐您查看这篇[文章](https://huggingface.co/blog/mask2former)，其中介绍了由Meta推出的新模型Mask2former，该模型仅使用全景数据集就能实现这三种分割类型。

### 现代方法：基于视觉Transformer的分割

您可能听说过U-Net，这是一个用于图像分割的流行网络。它设计了多个卷积层，工作在两个主要阶段：下采样阶段，将图像压缩以理解其特征；以及上采样阶段，将图像扩展回原始大小以进行详细分割。

计算机视觉曾经被卷积模型主导，但最近已转向视觉Transformer方法。例如，*[Segment anything model (SAM)](https://arxiv.org/abs/2304.02643)*，是Meta AI Research，FAIR在2023年4月推出的一个基于提示的流行模型。该模型基于Vision Transformer (ViT) 模型，旨在创建一个可提示的（即，您可以提供描述希望分割图像中内容的文字）分割模型，能够在新图像上实现零样本迁移。该模型的优势来自于在可用的最大数据集上进行训练，包括11亿个掩膜和1100万张图像。建议您尝试[Meta的演示](https://segment-anything.com/)中的几个图像，甚至更好的是可以在transformers中试用[模型](https://huggingface.co/ybelkada/segment-anything)。

以下是如何在transformers中使用模型的示例。首先，我们将初始化`mask-generation`管道。然后，将图像传递到管道中进行推理。

```python
from transformers import pipeline

pipe = pipeline("mask-generation", model="facebook/sam-vit-base", device=0)

raw_image = Image.open("path/to/image").convert("RGB")

masks = pipe(raw_image)
```

更多关于如何使用该模型的详细信息可以在[文档](https://huggingface.co/docs/transformers/main/en/model_doc/sam)中找到。

### 如何评估分割模型？

您现在已经了解了如何使用分割模型，但如何评估它呢？如前一节所示，分割主要是一个监督学习任务。这意味着数据集由图像及其对应的掩膜组成，这些掩膜作为真实值。可以使用一些指标来评估您的模型。最常见的指标包括：

- **交并比（IoU）或Jaccard指数**：是预测掩膜和真实掩膜的交集与并集的比率。IoU可以说是分割任务中最常用的指标。其优点在于对类别不平衡不太敏感，因此通常在开始建模时是一个不错的选择。

![IoU](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/iou.png)

- **像素准确率**：像素准确率计算为正确分类的像素数量与总像素数量的比率。尽管是一个直观的指标，但由于对类别不平衡敏感，可能会产生误导。

![像素准确率](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/pixel-accuracy.png)

- **Dice系数**：它是交集的两倍与预测掩膜和真实掩膜之和的比率。Dice系数只是预测与真实掩膜之间的重叠百分比。当您需要对重叠之间的微小差异敏感时，这是一个不错的选择。

![Dice系数](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/dice-coefficient.png)

## 资源与进一步阅读

- [Segment Anything Paper](https://arxiv.org/abs/2304.02643)
- [Fine-tuning Segformer 博客文章](https://huggingface.co/blog/fine-tune-segformer)
- [Mask2former 博客文章](https://huggingface.co/blog/mask2former)
- [Hugging Face上关于分割任务的文档](https://huggingface.co/docs/transformers/main/tasks/semantic_segmentation)
- 如果您希望更深入地了解该主题，我们推荐您查看斯坦福的[分割讲座](https://www.youtube.com/watch?v=nDPWywWRIRo)。