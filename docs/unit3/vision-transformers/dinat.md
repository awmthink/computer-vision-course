# Dilated Neighborhood Attention Transformer (DINAT)

![DINAT 结构图](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/dinat_images/dina_comparison.png)
## 结构概览

Dilated Neighborhood Attention Transformer (DiNAT) 是一种创新的分层视觉Transformer，旨在提升深度学习模型的性能，尤其是在视觉识别任务中的表现。与传统Transformer使用的自注意力机制不同，DiNAT引入了Dilated Neighborhood Attention (DiNA)，在无需增加计算量的情况下，将局部注意力机制扩展为稀疏的全局注意力。这一扩展使DiNA能够捕捉更多的全局上下文，指数级地扩大感受野，并有效地建模长距离依赖关系。

DiNAT在其架构中结合了NA和DiNA，从而创建了一个能够保持局部性、保持平移等变性，并在下游视觉任务中实现显著性能提升的Transformer模型。实验表明，与诸如NAT、Swin和ConvNeXt等强基线模型相比，DiNAT在各种视觉识别任务中表现出明显的优势。

## DiNAT 的核心：Neighborhood Attention

DiNAT 基于Neighborhood Attention (NA)架构，这是一种专门为计算机视觉任务设计的注意力机制，旨在高效地捕捉图像中像素之间的关系。简单来说，可以把它比作图像中每个像素需要理解并关注其周围像素，以更全面地理解整个图像。以下是NA的主要特性：

- **局部关系**：NA捕捉局部关系，使每个像素能够从其周围的邻域中获取信息。这类似于我们首先观察最近的物体来理解场景，然后再考虑整个视野的方式。

- **感受野**：NA允许像素扩展其对周围环境的理解，而无需增加过多计算量。它能够动态扩展像素的范围或“注意力范围”，在必要时将更远的邻居纳入其中。

总的来说，Neighborhood Attention是一种使图像中的像素能够专注于周围环境的技术，有助于高效地理解局部关系。这种局部化的理解在有效管理计算资源的同时，有助于构建对整个图像的详细理解。

![DiNAT 结构图](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/dinat_images/dinat_architecture.png)

## DINAT 的演变

Dilated Neighborhood Attention Transformer 的开发代表了视觉Transformer的一个重要改进。它解决了现有注意力机制的局限性。最初，Neighborhood Attention 被引入以提供局部性和效率，但在捕捉全局上下文方面有所不足。为了解决这个问题，引入了 Dilated Neighborhood Attention (DiNA) 概念。DiNA通过将邻域扩展为更大的稀疏区域，实现了更广泛的全局上下文捕获，且感受野成指数级增长，而不会增加计算负担。接下来的发展是DiNAT，它结合了局部的NA与DiNA扩展的全局上下文。DiNAT通过在模型中逐步更改扩张率，优化了感受野，简化了特征学习。

## 使用 DiNAT 进行图像分类
您可以使用[shi-labs/dinat-mini-in1k-224](https://huggingface.co/shi-labs/dinat-small-in1k-224)模型在ImageNet-1k图像中进行分类，或者根据自己的需求进行微调。

```python
from transformers import AutoImageProcessor, DinatForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoImageProcessor.from_pretrained("shi-labs/dinat-mini-in1k-224")
model = DinatForImageClassification.from_pretrained("shi-labs/dinat-mini-in1k-224")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# 模型预测1000个ImageNet类别中的一个
predicted_class_idx = logits.argmax(-1).item()
print("预测类别:", model.config.id2label[predicted_class_idx])
```

## 参考文献
- [DiNAT 论文](https://arxiv.org/abs/2209.15001) [1] 
- [Hugging Face DiNAT Transformer](https://huggingface.co/docs/transformers/model_doc/dinat) [2]  
- [Neighborhood Attention(NA)](https://arxiv.org/abs/2204.07143) [3] 
- [SHI Labs](https://huggingface.co/shi-labs) [4]  
- [OneFormer 论文](https://arxiv.org/abs/2211.06220) [5]
- [Hugging Face OneFormer](https://huggingface.co/docs/transformers/main/en/model_doc/oneformer) [6]
- [DiNAT Paper](https://arxiv.org/abs/2209.15001) [1] 
- [Hugging Face DiNAT Transformer](https://huggingface.co/docs/transformers/model_doc/dinat) [2]  
- [Neighborhood Attention(NA)](https://arxiv.org/abs/2204.07143) [3] 
- [SHI Labs](https://huggingface.co/shi-labs) [4]  
- [OneFormer Paper](https://arxiv.org/abs/2211.06220) [5]
- [Hugging Face OneFormer](https://huggingface.co/docs/transformers/main/en/model_doc/oneformer) [6]  
