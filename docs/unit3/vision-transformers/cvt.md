# 卷积视觉 Transformer (CvT)

在本节中，我们将深入探讨卷积视觉 Transformer (CvT)，这是一种视觉 Transformer (ViT) 的变体[[1]](#vision-transformer)，在计算机视觉的图像分类任务中被广泛应用。

## 回顾

在进入 CvT 之前，我们先简要回顾一下之前章节中讨论的 ViT 架构，以便更好地理解 CvT 架构。ViT 将每幅图像分解为具有固定长度的序列标记（即不重叠的图像块），然后应用多个标准的 Transformer 层，其中包括多头自注意力和位置前馈模块 (FFN)，以建模全局关系进行分类。

## 概述

卷积视觉 Transformer (CvT) 模型是 Haiping Wu、Bin Xiao、Noel Codella、Mengchen Liu、Xiyang Dai、Lu Yuan 和 Lei Zhang 在其论文《CvT：向视觉 Transformer 引入卷积》[[2]](#cvt) 中提出的。CvT 结合了 CNN 的所有优点：_局部感受野_、_共享权重_、_空间下采样_，以及 _平移_、_缩放_、_畸变不变性_，同时保留 Transformer 的优点：_动态注意力_、_全局上下文融合_、_更好的泛化能力_。与 ViT 相比，CvT 在保持计算效率的同时实现了更优的性能。此外，由于卷积引入了内建的局部上下文结构，CvT 不再需要位置嵌入，这使其在适应需要可变输入分辨率的广泛视觉任务方面具有潜在优势。

## 架构

![CvT 架构](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/cvt_architecture.png)
_(a) 整体架构，展示了通过卷积标记嵌入层实现的分层多阶段结构。 (b) 卷积 Transformer 块的详细信息，卷积投影作为第一层。[[2]](#cvt)_

上图展示了 CvT 架构的 3 阶段流水线的主要步骤。CvT 的核心在于将两种基于卷积的操作融合到视觉 Transformer 架构中：

- **卷积标记嵌入**：将输入图像分割为重叠的图像块，重组为标记，然后输入卷积层。这减少了标记数量（类似于下采样图像中的像素），同时增强其特征丰富度，类似于传统的 CNN。不像其他 Transformer，我们跳过为标记添加预定义的位置信息，而完全依赖卷积操作来捕获空间关系。

![投影层](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/cvt_conv_proj.png)
_(a) ViT 中的线性投影。 (b) 卷积投影。 (c) 压缩卷积投影（CvT 中的默认设置）。[[2]](#cvt)_

- **卷积 Transformer 块**：CvT 的每个阶段包含多个此类块。在此，我们使用深度可分离卷积（卷积投影）来处理自注意力模块的“查询”、“键”和“值”组件，而不是 ViT 中的线性投影，如上图所示。这保留了 Transformer 的优点，同时提高了效率。请注意，“分类标记”（用于最终预测）仅在最后一个阶段添加。最后，一个标准的全连接层对最终的分类标记进行分析，以预测图像类别。

### CvT 架构与其他视觉 Transformer 的比较

下表显示了上述代表性并行工作与 CvT 之间在位置编码的必要性、标记嵌入类型、投影类型和主干中的 Transformer 结构方面的关键差异。

| 模型                                            | 需要位置编码 (PE)            | 标记嵌入类型                     | 注意力投影类型            | 分层 Transformer         |
| ------------------------------------------------ | ---------------------------- | ------------------------------- | ------------------------ | ------------------------- |
| ViT[[1]](#vision-transformer), DeiT [[3]](#deit) | 是                           | 非重叠                           | 线性                     | 否                        |
| CPVT[[4]](#cpvt)                                 | 否 (带 PE 生成器)            | 非重叠                           | 线性                     | 否                        |
| TNT[[5]](#tnt)                                   | 是                           | 非重叠（图像块 + 像素）           | 线性                     | 否                        |
| T2T[[6]](#t2t)                                   | 是                           | 重叠（拼接）                     | 线性                     | 部分 (标记化)              |
| PVT[[7]](#pvt)                                   | 是                           | 非重叠                           | 空间缩减                 | 是                        |
| _CvT_[[2]](#cvt)                                 | _否_                         | _重叠（卷积）_                  | _卷积_                   | _是_                     |

### 主要亮点

CvT 实现卓越性能和计算效率的四个主要亮点如下：

- 包含新的 **卷积标记嵌入** 的 **分层 Transformer**。
- 利用 **卷积投影** 的卷积 Transformer 块。
- 由于卷积引入了内建的局部上下文结构， **不需要位置编码**。
- 相较于其他视觉 Transformer 架构，**参数更少**且 **FLOPs**（每秒浮点运算次数）更低。

## PyTorch 实现

现在是实践的时间！让我们探索如何在 PyTorch 中编写 CvT 架构的每个主要模块，代码基于官方实现 [[8]](#cvt-imp)。

1. 导入所需的库

```python
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
```

2. **卷积投影**的实现

```python
def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
    if method == "dw_bn":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            dim_in,
                            dim_in,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            bias=False,
                            groups=dim_in,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(dim_in)),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "avg":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "avg",
                        nn.AvgPool2d(
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            ceil_mode=True,
                        ),
                    ),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "linear":
        proj = None
    else:
        raise ValueError("Unknown method ({})".format(method))

    return proj
```

该方法接受与卷积层相关的多个参数（例如输入和输出维度、核大小、填充、步幅和方法），并根据指定的方法返回一个投影块。

- 如果方法是 `dw_bn`（深度可分离卷积与批量归一化），则会创建一个包含深度可分离卷积层、批量归一化层以及维度重排列的 Sequential 块。

- 如果方法是 `avg`（平均池化），则会创建一个包含平均池化层以及维度重排列的 Sequential 块。

- 如果方法是 `linear`，则返回 None，表示不应用任何投影。

维度的重排列是通过 `Rearrange` 操作来实现的，该操作对输入张量进行重塑。最终返回生成的投影块。

3. **卷积令牌嵌入**的实现

```python
class ConvEmbed(nn.Module):
    def __init__(
        self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x
```

此代码定义了一个 ConvEmbed 模块，用于对输入图像进行分块嵌入。

- `__init__` 方法初始化了模块的参数，例如 `patch_size`（图像块的大小）、`in_chans`（输入通道数）、`embed_dim`（嵌入块的维度）、`stride`（卷积操作的步幅）、`padding`（卷积操作的填充）和 `norm_layer`（归一化层，可选）。

- 在构造函数中，创建了一个 2D 卷积层（`nn.Conv2d`），并根据指定参数配置了块大小、输入通道、嵌入维度、步幅和填充。该卷积层分配给 `self.proj`。

- 如果提供了归一化层，则会创建一个包含嵌入维度通道的归一化层实例，并分配给 `self.norm`。

- `forward` 方法接收输入张量 `x`，并通过 `self.proj` 执行卷积操作。输出使用 `rearrange` 函数重塑以展平空间维度。如果存在归一化层，则对展平后的表示应用归一化。最后，张量被重塑回原始空间维度并返回。

总之，该模块旨在对图像进行分块嵌入，每个块独立通过卷积层处理，并对嵌入特征应用可选的归一化。

4. 实现 **Vision Transformer** 模块

```python
class VisionTransformer(nn.Module):
    """支持使用补丁或混合 CNN 输入阶段的 Vision Transformer"""

    def __init__(
        self,
        patch_size=16,
        patch_stride=16,
        patch_padding=0,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        with_cls_token = kwargs["with_cls_token"]
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # 随机深度衰减规则

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)

        if init == "xavier":
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

        def forward(self, x):
            x = self.patch_embed(x)
            B, C, H, W = x.size()

            x = rearrange(x, "b c h w -> b (h w) c")

            cls_tokens = None
            if self.cls_token is not None:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks):
                x = blk(x, H, W)

            if self.cls_token is not None:
                cls_tokens, x = torch.split(x, [1, H * W], 1)
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

            return x, cls_tokens
```

该代码定义了一个 Vision Transformer 模块。以下是代码的简要概述：

- **初始化：** `VisionTransformer` 类通过多种参数初始化，这些参数定义了模型结构，例如补丁大小、嵌入维度、层数、注意力头数、丢弃率等。

- **补丁嵌入：** 模型包含一个补丁嵌入层（`patch_embed`），用于通过卷积处理输入图像，将其划分为非重叠的补丁并进行嵌入。

- **Transformer 块：** 模型由多个 transformer 块（`Block`）组成。块的数量由深度参数控制。每个块包含多头自注意机制和前馈神经网络。

- **分类标记：** 模型可选包含一个可学习的分类标记（`cls_token`），该标记附加到输入序列中，常用于分类任务。

- **随机深度：** 随机深度应用于 transformer 块，训练期间随机跳过部分块以改善正则化效果。通过 `drop_path_rate` 参数控制。

- **权重初始化：** 模型权重使用截断正态分布（`trunc_norm`）或 Xavier 初始化（`xavier`）进行初始化。

- **前向方法：** 前向方法将输入通过补丁嵌入层处理，重排维度后（如果存在）添加分类标记，应用 dropout，然后通过 transformer 块堆栈。最后将输出重排回原始形状，并在返回输出前将分类标记（如果存在）与序列其他部分分离。

5. 实现卷积视觉 Transformer 模块（**Transformer 层次结构**）

```python
class ConvolutionalVisionTransformer(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        spec=None,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec["NUM_STAGES"]
        for i in range(self.num_stages):
            kwargs = {
                "patch_size": spec["PATCH_SIZE"][i],
                "patch_stride": spec["PATCH_STRIDE"][i],
                "patch_padding": spec["PATCH_PADDING"][i],
                "embed_dim": spec["DIM_EMBED"][i],
                "depth": spec["DEPTH"][i],
                "num_heads": spec["NUM_HEADS"][i],
                "mlp_ratio": spec["MLP_RATIO"][i],
                "qkv_bias": spec["QKV_BIAS"][i],
                "drop_rate": spec["DROP_RATE"][i],
                "attn_drop_rate": spec["ATTN_DROP_RATE"][i],
                "drop_path_rate": spec["DROP_PATH_RATE"][i],
                "with_cls_token": spec["CLS_TOKEN"][i],
                "method": spec["QKV_PROJ_METHOD"][i],
                "kernel_size": spec["KERNEL_QKV"][i],
                "padding_q": spec["PADDING_Q"][i],
                "padding_kv": spec["PADDING_KV"][i],
                "stride_kv": spec["STRIDE_KV"][i],
                "stride_q": spec["STRIDE_Q"][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)

            in_chans = spec["DIM_EMBED"][i]

        dim_embed = spec["DIM_EMBED"][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec["CLS_TOKEN"][-1]

        # 分类头
        self.head = (
            nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        )
        trunc_normal_(self.head.weight, std=0.02)

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f"stage{i}")(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x
```

该代码定义了一个名为 `ConvolutionalVisionTransformer` 的 PyTorch 模块。

- 模型包含多个阶段，每个阶段由 `VisionTransformer` 类的一个实例表示。
- 每个阶段具有不同的配置，如补丁大小、步幅、深度、头数等，由 spec 字典指定。
- `forward_features` 方法将输入 x 依次通过所有阶段，并聚合最终表示。
- 该类具有一个分类头，用于执行线性变换以生成最终输出。
- `forward` 方法调用 `forward_features`，然后将结果传递到分类头。
- 视觉 Transformer 阶段按顺序命名为 stage0、stage1 等，每个阶段都是 `VisionTransformer` 类的实例，形成 Transformer 的层次结构。

恭喜！现在你已经了解了如何在 PyTorch 中实现 CvT 架构。你可以在 [这里](https://github.com/microsoft/CvT/blob/main/lib/models/cls_cvt.py) 查看完整的 CvT 架构代码。

## 试试看

如果你想使用 CvT 而不深入了解其 PyTorch 实现的复杂细节，可以通过 Hugging Face 的 `transformers` 库轻松使用。操作如下：

```bash
pip install transformers
```

你可以在 [这里](https://huggingface.co/docs/transformers/model_doc/cvt#overview) 查找 CvT 模型的文档。

### 用法

以下是如何使用 CvT 模型将 COCO 2017 数据集中的一张图像分类为 1,000 个 ImageNet 类别之一：

```python
from transformers import AutoFeatureExtractor, CvtForImageClassification
from PIL

 import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/cvt-13")
model = CvtForImageClassification.from_pretrained("microsoft/cvt-13")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# 模型预测 1,000 个 ImageNet 类别中的一个
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

## 参考资料

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) <a id="vision-transformer"></a>
- [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808) <a id="cvt"></a>
- [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) <a id="deit"></a>
- [Conditional Positional Encodings for Vision Transformers](https://arxiv.org/abs/2102.10882) <a id="cpvt"></a>
- [Transformer in Transformer](https://arxiv.org/abs/2103.00112v3)<a id="tnt"></a>
- [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986) <a id="t2t"></a>
- [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122) <a id="pvt"></a>
- [Implementation of CvT](https://github.com/microsoft/CvT/tree/main) <a id="cvt-imp"></a>
