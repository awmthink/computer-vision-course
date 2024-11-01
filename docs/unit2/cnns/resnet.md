# ResNet（残差网络）

神经网络随着层数增加而被认为更加有效，因为增加层数可以提升模型的表现。

随着网络的加深，提取的特征可以得到进一步丰富，例如VGG16和VGG19。

于是产生了一个疑问：“网络学习是否如简单堆叠更多层一样容易？”
为了解决这个问题中的一个障碍——梯度消失问题，采用了归一化的初始化方式和中间归一化层。

然而，一个新的问题出现了：退化问题。随着神经网络变得更深，准确度趋于饱和并快速退化。一项对比浅层和深层平面网络的实验表明，深层模型在训练和测试中表现出更高的错误率，表明在有效训练深层架构方面存在根本性的挑战。这种退化不是由于过拟合，而是因为网络变深时训练误差增加。增加的层未能逼近身份函数。

ResNet的残差连接释放了极深度网络的潜力，相较于先前架构显著提升了准确性。

## ResNet架构

- 一个残差模块。来源：ResNet论文  

![residual](https://huggingface.co/datasets/hf-vision/course-assets/blob/main/ResnetBlock.png)

ResNet的构建块被设计为身份函数，在保留输入信息的同时进行学习。这种方法确保了高效的权重优化，并防止了网络加深时的退化。

ResNet的构建模块如图所示，来源：ResNet论文。

![resnet_building_block](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/ResnetBlock.png)

快捷连接执行身份映射，其输出会加到堆叠层的输出上。身份快捷连接既不增加额外的参数也不增加计算复杂度，这些连接绕过层级，为信息流动创造了直接路径，使得神经网络能够学习残差函数 \(F\)。

我们可以将ResNet网络总结为 -> 平面网络 + 快捷连接！

对于操作 \(F(x) + x\)，\(F(x)\) 和 \(x\) 应具有相同的维度。
ResNet采用了两种技术来实现这一点：

- 零填充快捷连接：添加全为零的通道来保持维度，同时不引入额外需要学习的参数。
- 投影快捷连接：使用1x1卷积在必要时调整维度，包含一些可学习的额外参数。

在更深的ResNet架构中，例如ResNet 50、101和152，采用了一种专门的“瓶颈构建块”来管理参数复杂性，保持效率，同时允许更深度的学习。

## ResNet代码

### ImageNet上预训练的深度残差网络
以下展示如何使用transformers库加载带有图像分类头的预训练ResNet模型。
```python
from transformers import ResNetForImageClassification

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

model.eval()
```
所有预训练模型要求输入图像按照相同的方式进行归一化，即具有3通道的RGB图像的小批次，形状为(3 x H x W)，其中H和W至少为224。图像需加载至[0, 1]范围内，然后使用mean = [0.485, 0.456, 0.406]和std = [0.229, 0.224, 0.225]进行归一化。

以下为一个示例执行过程。该示例可以在[Hugging Face文档](https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/resnet)中找到。

```python
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# 模型预测1000个ImageNet类别之一
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

## 参考文献

- [PyTorch文档](https://pytorch.org/hub/pytorch_vision_resnet/)
- [ResNet: 深度残差学习用于图像识别](https://arxiv.org/abs/1512.03385)
- [Resnet架构来源：ResNet论文](https://arxiv.org/abs/1512.03385)
- [Hugging Face关于ResNet的文档](https://huggingface.co/docs/transformers/en/model_doc/resnet)