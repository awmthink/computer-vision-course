# 检测转换器 (DETR)
## 架构概述
DETR 主要用于目标检测任务，即在图像中检测对象。例如，模型的输入可以是一条道路的图像，模型的输出可能是 `[('car',X1,Y1,W1,H1),('pedestrian',X2,Y2,W2,H2)]`，其中 X、Y、W、H 分别表示边界框的位置（x、y 坐标）以及框的宽度和高度。  
传统的目标检测模型如 YOLO 包含手工设计的特征，如锚框先验，这需要对对象位置和形状进行初始猜测，从而影响后续的训练。随后，还需进行后处理步骤以移除重叠的边界框，这需要仔细选择过滤启发式方法。  
检测转换器（简称 DETR）通过在特征提取骨干网络后使用编码器-解码器转换器，简化了检测器的结构，直接并行预测边界框，几乎不需要后处理。 

![DETR 架构图](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/DETR.png)  
DETR 的模型架构始于 CNN 骨干网络，类似于其他基于图像的网络，其输出经过处理后被输入到转换器编码器，生成 N 个嵌入。编码器嵌入添加了学习到的位置嵌入（称为对象查询）并用于转换器解码器中，生成另一个 N 个嵌入。最后步骤中，每个 N 个嵌入都通过独立的前馈层来预测边界框的宽度、高度、坐标以及对象类别（或是否存在对象）。

## 关键特性

### 编码器-解码器
与其他转换器类似，转换器编码器要求 CNN 骨干网络的输出是一个序列。因此，特征图的尺寸 `[dimension, height, width]` 被缩小并展平为 `[dimension, less than height x width]`。  
![编码器的特征图](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/DETR_FeatureMaps.png)  
_**左**：特征图中的 256 维度中的 12 个维度可视化。每个维度提取了原始猫图像的一些特征，同时缩小了原图。一些维度更加关注猫的图案；另一些则更关注床单的图案。_  
_**右**：保持特征维度为 256 的原始大小，宽度和高度进一步缩小并展平成大小为 850。_  
由于转换器是置换不变的，因此在编码器和解码器中加入了位置嵌入，以提醒模型嵌入来自图像的哪个位置。在编码器中，使用了固定的位置编码，而在解码器中，使用了学习到的位置编码（对象查询）。固定编码类似于原始转换器论文中使用的编码，其中编码由在不同特征维度上具有不同频率的正弦函数定义。它无需任何学习参数即可传达位置感，由图像上的位置索引。学习编码也由位置索引，但每个位置都有一个独立的编码，学习过程中进行调整，以使模型理解位置。

### 基于集合的全局损失函数
在 YOLO（一个流行的目标检测模型）中，损失函数包括边界框、对象性（即对象存在于兴趣区域的概率）和类别损失。损失在每个网格单元的多个边界框上计算，这是一个固定数目。另一方面，在 DETR 中，架构期望生成独特的边界框，以置换不变的方式输出（即，输出的检测顺序不重要，边界框必须变化，不能都相同）。因此，需要匹配来评估预测的优劣。

**二分匹配**  
二分匹配是一种计算地面真实边界框与预测框之间一对一匹配的方法。它找到地面真实边界框和预测框之间最高相似度的匹配，包括类别。这样可以确保最接近的预测与相应的地面真实框匹配，以便在损失函数中正确调整边界框和类别。如果不进行匹配，即使预测顺序不与地面真实框对齐，也会被标记为错误。

## 使用 DETR 进行目标检测
要查看如何使用 Hugging Face 转换器对 DETR 进行推理的示例，请参见 `DETR.ipynb`。

## DETR 的演进
### 变形 DETR
DETR 的两个主要问题是收敛过程长且缓慢，以及对小物体检测效果欠佳。  
**变形注意力**  
第一个问题通过使用变形注意力解决，它减少了需要关注的采样点数量。传统注意力由于全局注意力效率低下，限制了图像的分辨率。模型仅关注每个参考点周围的固定数量采样点，参考点基于输入由模型学习。例如，在狗的图像中，一个参考点可能位于狗的中心，采样点则分布在耳朵、嘴巴、尾巴等周围。  

**多尺度变形注意力模块**  
第二个问题类似于 YOLOv3 的解决方案，即引入多尺度特征图。在卷积神经网络中，较早层提取较小的细节（如线条），而较晚层提取较大的细节（如车轮、耳朵）。类似地，不同层次的变形注意力导致不同分辨率的特征图。通过将编码器某些层的输出连接到解码器，允许模型检测多种尺寸的对象。

### 条件 DETR
条件 DETR 也致力于解决原始 DETR 的慢速训练收敛问题，实现了超过 6.7 倍的加速收敛。作者发现对象查询是通用的，并不专属于输入图像。使用解码器中的 **条件交叉注意力**，查询可以更好地定位边界框回归的区域。  
![变形 DETR 解码器层](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/DETR_DecoderLayer.png)  
_左：DETR 解码器层。右：变形 DETR 解码器层_  
上述图中比较了原始 DETR 和变形 DETR 的解码器层，主要区别在于交叉注意力块的查询输入。作者在内容查询 c<sub>q</sub>（解码器自注意力输出）和空间查询 p<sub>q</sub> 之间做了区分。原始 DETR 仅简单地将它们相加。在变形 DETR 中，它们被拼接，c<sub>q</sub> 聚焦于对象的内容，而 p<sub>q</sub> 聚焦于边界框区域。  
空间查询 p<sub>q</sub> 是解码器嵌入和对象查询同时投射到相同空间的结果（分别成为 T 和 p<sub>s</sub>），然后两者相乘。前几层的解码器嵌入包含边界框区域的信息，而对象查询包含每个边界框的参考点信息。因此，它们的投影结合成一种表示，允许交叉注意力通过与编码器输入和正弦位置嵌入的相似度来测量。这比仅使用对象查询和固定参考点的 DETR 更加有效。

## DETR 推理

您可以通过以下代码在 Hugging Face Hub 上使用现有的 DETR 模型进行推理：

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 初始化模型
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)

# 预处理输入并进行推理
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 将输出（边界框和类别 logits）转换为 COCO API 格式
# 非最大抑制阈值设为0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

输出如下。

```bash 
Detected cat with confidence 0.998 at location [344.06, 24.85, 640.34, 373.74]
Detected remote with confidence 0.997 at location [328.13, 75.93, 372.81, 187.66]
Detected remote with confidence 0.997 at location [39.34, 70.13, 175.56, 118.78]
Detected cat with confidence 0.998 at location [15.36, 51.75, 316.89, 471.16]
Detected couch with confidence 0.995 at location [-0.19, 0.71, 639.73, 474.17]
```

## DETR 的 PyTorch 实现

以下是根据原始论文实现的 DETR：

```python
import torch
from torch import nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(
        self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers
    ):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        h = self.transformer(
            pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1)
        )
        return self.linear_class(h), self.linear_bbox(h).sigmoid()
```
### `forward`函数逐行分析：

**Backbone**   
输入图像首先经过 ResNet Backbone 和一个卷积层，将维度降低到 `hidden_dim`。
```python
x = self.backbone(inputs)
h = self.conv(x)
```
它们在 `__init__` 函数中声明。
```python
self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
self.conv = nn.Conv2d(2048, hidden_dim, 1)
```

**位置嵌入**

虽然在论文中固定和训练的嵌入分别用于编码器和解码器，但在实现中作者为了简化将训练的嵌入用于两者。
```python
pos = (
    torch.cat(
        [
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ],
        dim=-1,
    )
    .flatten(0, 1)
    .unsqueeze(1)
)
```
在此声明为 `nn.Parameter`。行和列嵌入的组合表示图像中的位置。
```python
self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
```

**调整大小**   
在进入 Transformer 之前，大小为 `(batch size, hidden_dim, H, W)` 的特征被调整为 `(hidden_dim, batch size, H*W)`，使其成为 Transformer 的序列输入。
```python
h.flatten(2).permute(2, 0, 1)
```

**Transformer**   
`nn.Transformer` 函数的第一个参数是传递给编码器的输入，第二个参数是传递给解码器的输入。如您所见，编码器接收调整大小后的特征加上位置嵌入，而解码器接收 `query_pos`，即解码器的位置嵌入。
```python
h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
```

**前馈网络**   
最后，输出是一个大小为 `(query_pos_dim, batch size, hidden_dim)` 的张量，通过两个线性层处理。
```python
return self.linear_class(h), self.linear_bbox(h).sigmoid()
```
第一个线性层预测类别，并增加了一个 `No Object` 类。
```python
self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
```
第二个线性层预测边界框，输出大小为4，用于表示 xy 坐标、宽度和高度。
```python
self.linear_bbox = nn.Linear(hidden_dim, 4)
```


## 参考

- [DETR](https://arxiv.org/abs/2005.12872) 
- [YOLO](https://arxiv.org/abs/1506.02640) 
- [YOLOv3](https://arxiv.org/abs/1804.02767) 
- [Conditional DETR](https://arxiv.org/abs/2108.06152) 
- [Deformable DETR](https://arxiv.org/abs/2010.04159) 
- [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50)
