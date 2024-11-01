# MobileViT v2

前面讨论的 Vision Transformer 架构计算密集，难以在移动设备上运行。以往的移动视觉任务的最先进架构使用了 CNN，然而，CNN 无法学习全局表示，因此其性能不如 Transformer。

MobileViT 架构旨在解决视觉移动任务的需求问题，如低延迟和轻量化架构，同时提供 Transformer 和 CNN 的优势。MobileViT 架构由 Apple 开发，并建立在 Google 研究团队的 MobileNet 基础上。MobileViT 架构在先前的 MobileNet 架构上增加了 MobileViT Block 和可分离的自注意力。这两个特性使其实现了超快的延迟、参数和计算复杂度的降低，以及在资源受限设备上部署视觉 ML 模型的能力。

## MobileViT 架构

Sachin Mehta 和 Mohammad Rastegari 在论文《MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer》中提出的 MobileViT 架构如下所示：
![MobileViT Architecture](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-Architecture.png)

部分内容与前一章内容类似。包括 MobileNet 块、nxn 卷积、下采样、全局池化和最终的线性层。

从全局池化层和线性层可以看出，这里展示的模型用于分类。然而，本文中引入的相同模块可用于多种视觉应用。

## MobileViT 块

MobileViT 块结合了 CNN 的局部处理和 Transformer 的全局处理。它结合了卷积和 Transformer 层，能够捕获空间局部信息和数据的全局依赖关系。

MobileViT 块的示意图如下：
![MobileViT Block](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-MobileViTBlock.png)

内容较多，让我们逐步分解。

- 该块接收多通道图像。假设对于 RGB 图像有 3 个通道，因此该块接收一个三通道图像。
- 然后对这些通道执行 N x N 卷积，将结果附加到现有通道上。
- 随后该块对这些通道进行线性组合，并将它们添加到现有的通道堆栈中。
- 对于每个通道，这些图像被展开为扁平化的补丁。
- 然后这些扁平化补丁通过 Transformer，以生成新的补丁。
- 这些补丁随后被重新组合为一个具有 d 维的图像。
- 然后在拼接后的图像上叠加一个逐点卷积。
- 最后，拼接图像与原始 RGB 图像重新组合。

这种方法允许在 H x W（整个输入大小）上拥有接收场，同时通过保持补丁的位置信息来建模非局部和局部依赖关系。这可以通过补丁的展开和重新组合来实现。

<Tip>
接收场是输入空间中影响特定层特征的区域大小。
</Tip>

这种复合方法使 MobileViT 的参数数量比传统 CNN 更少，且准确性更高！
![MobileViT CNNPreformance](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-CNNPreformance.png)

原始 MobileViT 架构的主要效率瓶颈在于 Transformer 中的多头自注意力，其相对于输入标记的时间复杂度为 O(k^2)。

多头自注意力还需要耗费资源的批量矩阵乘法操作，这在资源受限的设备上会影响延迟。

这些作者在另一篇论文中提出了一种加速注意力操作的方法，称为可分离自注意力。

## 可分离自注意力

在传统的多头注意力中，相对于输入标记的复杂度是二次的（O(k^2)）。该文中提出的可分离自注意力相对于输入标记的复杂度为 O(k)。

此外，这种注意力方法不使用任何批量矩阵乘法，这有助于减少在移动电话等资源受限设备上的延迟。

这是一个巨大的改进！

<Tip>
已有许多不同形式的注意力，其复杂度范围从 O(k) 到 O(k*sqrt(k))，再到 O(k*log(k))。

可分离自注意力并不是第一个实现 O(k) 复杂度的。在 [Linformer](https://arxiv.org/abs/2006.04768) 中，Attention 的 O(k) 复杂度也得以实现。

但它仍然使用了昂贵的批量矩阵乘法。
</Tip>

Transformer、Linformer 和 MobileViT 的注意力机制对比如下所示：
![Attention Comparison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-Attention.png)

上图显示了 Transformer、Linformer 和 MobileViT v2 架构中各类注意力机制的对比。

例如，在 Transformer 和 Linformer 架构中，注意力计算需要进行两次批量矩阵乘法。

而在可分离自注意力的情况下，这两次批量矩阵乘法被替换为两次独立的线性计算，从而进一步提高了推理速度。

## 结论

MobileViT 块在保持空间局部信息的同时，开发了全局表示，结合了 Transformer 和 CNN 的优势。它提供了一个涵盖整个图像的接收场。

将可分离自注意力引入现有架构，进一步提高了准确性和推理速度。
![Inference Tests](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MobileViT-Inference.png)

在 iPhone 12s 上进行的不同架构测试显示，引入可分离注意力后，性能显著提升，如上图所示！

总体而言，MobileViT 架构是一个非常强大的资源受限视觉任务架构，提供快速的推理速度和高准确性。

## Transformers 库

如果您想在本地尝试 MobileViTv2，可以从 HuggingFace 的 `transformers` 库中使用，方法如下：

```bash
pip install transformers
```
以下是如何使用 MobileViT 模型对图像进行分类的简短示例。

```python
from transformers import AutoImageProcessor, MobileViTV2ForImageClassification
from datasets import load_dataset
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained(
    "apple/mobilevitv2-1.0-imagenet1k-256"
)
model = MobileViTV2ForImageClassification.from_pretrained(
    "apple/mobilevitv2-1.0-imagenet1k-256"
)

inputs = image_processor(image, return_tensors="pt")

logits = model(**inputs).logits

# 模型预测 1000 个 ImageNet 类别中的一个
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

## 推理 API

为了实现更轻量的计算机视觉设置，您可以使用 Hugging Face 推理 API 与 MobileViTv2 进行交互。推理 API 是一种与 Hugging Face Hub 上的多个模型交互的 API。
我们可以通过 Python 如下查询推理 API。

```py
import json
import requests

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = (
    "https://api-inference.huggingface.co/models/apple/mobilevitv2-1.0-imagenet1k-256"
)


def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


data = query("cats.jpg")
```
我们可以使用 JavaScript 以相同方式进行查询，如下所示。
```js
import fetch from "node-fetch";
import fs from "fs";
async function query(filename) {
    const data = fs.readFileSync(filename);
    const response = await fetch(
        "https://api-inference.huggingface.co/models/apple/mobilevitv2-1.0-imagenet1k-256",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "POST",
            body: data,
        }
    );
    const result = await response.json();
    return result;
}
query("cats.jpg").then((response) => {
    console.log(JSON.stringify(response));
});
```
最后，我们可以通过 curl 查询推理 API。
```bash
curl https://api-inference.huggingface.co/models/apple/mobilevitv2-1.0-imagenet1k-256 \
        -X POST \
        --data-binary '@cats.jpg' \
        -H "Authorization: Bearer ${HF_API_TOKEN}"
```