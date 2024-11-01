# 视觉中的 Retention

## 什么是保留网络
保留网络（RetNet）是为大语言模型提出的一种基础架构，该架构在论文 [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) 中被首次提出。RetNet 的设计旨在解决大规模语言建模中的关键问题：训练的并行性、低成本推理和优异的性能。

![LLM Challenges](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/LLM%20Challenges.png)
RetNet 通过引入多尺度保留（Multi-Scale Retention, MSR）机制应对这些挑战，这是对 Transformer 模型中多头注意力机制的替代方案。
MSR 具备双重的递归性和并行性，因此可以在并行训练的同时进行递归推理。我们将在后续章节中详细探讨 RetNet。

多尺度保留机制在以下三种计算范式下运行：
- **并行表示：** RetNet 的这一方面类似于 Transformer 中的自注意力机制，使得我们能够高效地在 GPU 上训练模型。

- **递归表示：** 此表示通过 $O(1)$ 的内存和计算复杂度实现高效推理，显著降低了部署成本和延迟，并简化了实现，不需要传统模型中常用的键值缓存策略。

- **块级递归表示：** 第三种范式针对长序列建模的挑战，通过并行编码每个局部块来提高计算速度，同时递归编码全局块以优化 GPU 内存使用。

在训练阶段，该方法结合了并行和块级递归表示，优化 GPU 使用以实现快速计算，在长序列的计算效率和内存使用方面尤为有效。
在推理阶段，采用递归表示，支持自回归解码。该方法有效地减少了内存使用和延迟，同时保持了等效的性能表现。

## 从语言到图像
### RMT
论文 [RMT: Retentive Networks Meet Vision Transformers](https://arxiv.org/abs/2309.11523) 提出了一种受 RetNet 架构启发的新型视觉主干网络。作者提出 RMT，通过引入显式空间先验并减少计算复杂度，增强 Vision Transformer (ViT)，灵感来源于 RetNet 的并行表示。
包括将 RetNet 的时间衰减机制适配到空间域，并使用基于[曼哈顿距离](https://en.wikipedia.org/wiki/Taxicab_geometry)的空间衰减矩阵，以及一种分解的注意力形式来提高视觉任务的效率和可扩展性。

- **曼哈顿自注意力（MaSA）**
![Attention Comparison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Attention%20Comparison.png)
MaSA 结合了自注意力机制，基于曼哈顿距离在标记之间引入了二维双向空间衰减矩阵。该矩阵根据与目标标记的距离降低注意力分数，使模型能够在感知全局信息的同时根据距离变化调整注意力。

- **分解的曼哈顿自注意力（MaSAD）**
![MaSAD](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MaSAD.png)
该机制沿图像的水平和垂直轴分解自注意力，保持空间衰减矩阵的同时不丢失先验信息。这种分解方式使曼哈顿自注意力 (MaSA) 能以线性复杂度高效建模全局信息，同时保留了原始 MaSA 的感受野形状。

然而，与最初的 RetNet 在训练中使用并行表示、在推理中使用递归表示不同，RMT 在训练和推理中都采用 MaSA 机制。作者对 MaSA 与其他 RetNet 表示方式进行了对比，结果显示 MaSA 具有最高的吞吐量和最佳的准确性。
![MaSA vs Retention](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MaSA%20vs%20Retention.png)

### ViR
![ViR](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/ViR.png)

另一项受 RetNet 架构启发的研究是 ViR，详见论文 [ViR: Vision Retention Networks](http://arxiv.org/abs/2310.19731)。在该架构中，作者提出了一种通用的视觉主干网络，重新设计了保留机制。通过利用保留网络的并行与递归双重特性，作者证明了 ViR 在图像吞吐量和内存消耗方面可以较好地扩展到更高的图像分辨率。

ViR 的整体架构与 ViT 非常相似，但其将多头注意力 (MHA) 替换为多头保留 (MHR)。这种 MHR 机制不依赖任何门控功能，并且可以在并行、递归或块级（并行和递归的混合）模式之间切换。ViR 的另一项区别在于其首先将位置嵌入加到块嵌入上，然后再附加 [class] 标记。

## 拓展阅读

- [RetNet 的官方仓库](https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py)
- [RetNet 多尺度保留机制的官方仓库](https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiscale_retention.py)
- [Retentive Networks (RetNet) 解释：备受期待的 Transformer 替代品现已发布](https://medium.com/ai-fusion-labs/retentive-networks-retnet-explained-the-much-awaited-transformers-killer-is-here-6c17e3e8add8)
- [Retentive Network: A Successor to Transformer for Large Language Models (Paper Explained)](https://www.youtube.com/watch?v=ec56a8wmfRk)
- [RMT 的官方仓库](https://github.com/qhfan/RMT)
- [ViR 的官方仓库](https://github.com/NVlabs/ViR)