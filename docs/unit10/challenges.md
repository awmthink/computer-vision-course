# 使用合成数据的挑战和机遇

训练机器学习模型需要大量数据。合成数据可以帮助解决隐私问题、增强有限数据集以及纠正真实数据中的不平衡。在课程中，我们学习了多种生成合成数据的方法。然而，在使用合成数据来训练模型之前，需要考虑一些重要的因素。

## 模型过拟合

过拟合指的是机器学习模型过度学习了训练数据，以至于在新的、未见过的数据上表现不佳。这就像掌握了解决特定问题的方法，但遇到新的情况时这种策略却不起作用。如果生成合成数据的过程过于简单或存在过于一致的模式，那么模型可能会对合成数据中的有限变化过拟合。例如，假设你用一个包含25个红色圆形和25个蓝色方形的合成数据集训练了一个模型。模型很可能会学会将圆形与红色关联、方形与蓝色关联。当出现一个红色方形时，模型可能会出错。

<Tip warning="true">
  请确保你的数据集不存在以下这些模式！
</Tip>

_过于一致的颜色_
![consistent-color](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/overfit-color.jpg)

_过于一致的大小_
![consistent-size](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/overfit-size.jpg)

_过于一致的背景_
![consistent-background](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/overfit-background.jpg)

_过于一致的位置_
![consistent-location](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/overfit-location.jpg)

## 合成数据中是否存在偏差？

如果生成合成数据的过程存在偏差或不准确，模型可能会无意中学习并延续这些偏差。需要注意以下几个问题：

**多样性有限**

一个挑战在于合成数据可能无法充分代表真实数据的复杂性和多样性。虽然形状示例看似简单，但在实际应用中，未能考虑到人、地点、动物或物体的多样性会导致模型表现不佳。例如，假设你想训练一个模型来监测濒危物种（如指猴）的人口。如果你的数据集中只包含环尾狐猴的图像，那么模型可能难以在野外准确识别指猴。这种局限性可能导致种群评估中的误差。好消息是，如果你注意到基础数据集中存在的不平衡，可以通过使用合成数据来为缺失类别生成数据，从而去偏。

<Tip>
  尽量确保你的数据集反映出真实世界的多样性！
</Tip>

**丰富的多样性**
![nice-variety](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-overfit/good-variety.jpg)

**复制已有偏差**

如果你用来生成合成图像的数据本身存在偏差，模型可能会无意中学习并复制这些偏差。这就像抄袭朋友的笔记却没有意识到他们写错了内容——计算机可能会重复这些错误。

## 使用合成数据的好处是否超过计算成本？

生成高质量的合成数据可能需要较高的计算成本。这在时间和资源方面可能会带来挑战，尤其是对于复杂的模型或大数据集。一般来说，只有当生成和使用合成数据集能节省资源（资金、时间等）时，才有实际意义。

### 合成图像的感知质量如何？

让我们来看看用 DCGAN 生成的肺部图像。有些图像看起来很真实，而有些则不尽如人意。用低质量图像训练的模型可能无法检测到肺炎，因为这些图像中包含了真实图像中没有的噪声。也有可能你的模型会非常擅长识别合成数据中的模式，但这些模式在现实世界中可能不存在或不同。

一个好的做法是使用如 Frechet Inception Distance (FID)、Inception Score (IS) 或 Classification Accuracy Score (CAS) 等指标来评估你的数据集。

_FID：_

FID 使用一个预训练的神经网络模型，通常是 [Inception](https://huggingface.co/docs/timm/models/inception-v4)，该模型擅长识别图像中的对象。该模型用于从真实图像和生成图像中提取特征。FID 是一种测量两个分布之间距离的方法，考虑了分布的均值和协方差。

低 FID 表明真实图像和生成图像的特征分布相似，生成的图像更有可能是逼真的。

_IS：_

IS 使用预训练的 Inception 模型来评估生成图像的质量，特别适用于生成对抗网络（GAN）。对于每个生成的图像，Inception 模型根据其识别图像中对象的信心分配分数。高分数表示 Inception 模型对图像内容的识别较为自信。

_CAS：_

分类准确率是另一种衡量模型在合成数据上的表现的指标。较高的准确率表明模型能够有效地捕捉真实图像的特征和模式。对于某些类别较低的准确率可能表明生成过程存在问题，例如不真实的背景、错误的纹理或不一致的光照条件。你可以利用 CAS 来识别和解决这些问题，以提高合成数据集的整体质量。

## 结论

即使在训练完模型后，持续监测其在实际场景中的表现仍然至关重要。如果模型遇到合成数据中没有的新情况或趋势，它可能会遇到适应性问题。应对这些挑战需要在设计合成数据生成过程时慎重考虑，并评估模型在真实数据上的表现。应用这些原则将有助于释放合成数据的潜力！

## 资源和进一步阅读

- [Analyzing Effects of Fake Training Data on the Performance of Deep Learning Systems](https://arxiv.org/pdf/2303.01268.pdf)
- [Bridging the Gap: Enhancing the Utility of Synthetic Data Via Post-Processing Techniques](https://arxiv.org/pdf/2305.10118.pdf)
- [CIFAKE: Image Classification and Explanable Identification of AI-Generated Synthetic Images](https://arxiv.org/pdf/2303.14126.pdf)
- [Classification Accuracy Score for Conditional Generative Models](https://arxiv.org/abs/1905.10887)
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
- [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567v3.pdf)
- [Metrics](https://github.com/huggingface/community-events/tree/main/huggan/pytorch/metrics)
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid)