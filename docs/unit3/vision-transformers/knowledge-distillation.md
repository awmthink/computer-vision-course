# 使用视觉Transformer进行知识蒸馏

我们将学习知识蒸馏，这是一种支持 [distilGPT](https://huggingface.co/distilgpt2) 和 [distilbert](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) 等模型的方法——它们是 *Hugging Face Hub上下载量最高的模型之一！*

我们可能都遇到过一些“教学”方式是仅仅给出正确答案，然后测试我们从未见过的问题的老师，这类似于机器学习模型中的监督学习，我们为其提供标注数据集进行训练。然而，知识蒸馏（[Knowledge Distillation](https://arxiv.org/abs/1503.02531)）是一种不同的方法，可以让我们获得一个更小的模型，其性能可以媲美大型模型，并且速度更快。

## 知识蒸馏的直观理解

假设你被给了这样一道多项选择题：

![多项选择题](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multiple-choice-question.png)

如果有人直接告诉你“答案是Draco Malfoy”，这并不能教你太多关于每个角色与Harry Potter之间的关系。

另一方面，如果有人告诉你：“我非常肯定它不是Ron Weasley，我有点肯定它不是Neville Longbottom，并且我非常肯定它*是*Draco Malfoy”，这就为你提供了关于这些角色与Harry Potter之间关系的信息！这正是知识蒸馏范式下传递给学生模型的信息类型。

## 神经网络中的知识蒸馏

在论文[*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531)中，Hinton等人介绍了一种被称为知识蒸馏的训练方法，其灵感来源于*昆虫*。就像昆虫从幼虫变为适合不同任务的成虫一样，大规模机器学习模型最初就像幼虫一样繁重，用于从数据中提取结构信息，但可以将知识蒸馏到更小、更高效的模型中以便部署。

知识蒸馏的核心是使用教师网络的预测logits来向更小、更高效的学生模型传递信息。我们通过重新定义损失函数，使其包含*蒸馏损失*，以鼓励学生模型的输出分布接近教师模型的分布。

蒸馏损失的公式为：

![蒸馏损失](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/KL-Loss.png)

KL损失指的是[相对熵](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)在教师和学生的输出分布之间的差异。学生模型的总体损失是此蒸馏损失和标准交叉熵损失的和。

要查看此损失函数的Python实现和完整示例，请查看此部分的[笔记本](https://github.com/johko/computer-vision-course/blob/main/notebooks/Unit%203%20-%20Vision%20Transformers/KnowledgeDistillation.ipynb)。

<a target="_blank" href="https://colab.research.google.com/github/johko/computer-vision-course/blob/main/notebooks/Unit%203%20-%20Vision%20Transformers/KnowledgeDistillation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 利用知识蒸馏用于边缘设备

随着AI模型在边缘设备上的部署，知识蒸馏变得越来越重要。部署一个大小为1GB、延迟为1秒的大规模模型对于实时应用来说不可行，因为其高计算和存储需求。主要的限制来源于模型的大小。因此，知识蒸馏得到了广泛应用，这种技术可以在性能几乎不变的情况下将模型参数减少90%以上。

## 知识蒸馏的影响（优点与缺点）

### 1. 熵增益
在信息理论中，熵类似于物理中的“混乱”或无序度。在这里，它度量了分布包含的信息量。请看以下示例：

- 哪个更难记住：`[0, 1, 0, 0]` 还是 `[0.2, 0.5, 0.2, 0.1]`？

第一个向量 `[0, 1, 0, 0]` 更容易记住和压缩，因为它包含的信息更少，可以表示为“在第二位置为1”。而 `[0.2, 0.5, 0.2, 0.1]` 则包含更多信息。
进一步举例，假设我们用ImageNet训练了一个80M参数的网络，然后将其蒸馏为一个5M参数的学生模型。我们会发现教师模型的输出熵远低于学生模型的输出熵。这意味着，尽管学生模型的输出是正确的，但比起教师的输出更为混乱。这归因于教师的额外参数，使其在区分类别时提取更多特征。这种知识蒸馏的观点很有趣，当前的研究正在探讨如何通过使用熵作为损失函数或应用类似物理学的指标（例如能量）来减少学生的熵。

### 2. 连贯的梯度更新
模型通过最小化损失函数并通过梯度下降迭代学习参数更新。考虑参数集 `P = {w1, w2, w3, ..., wn}`，在教师模型中用于检测属于类别A的样本。如果某个模糊样本看起来像类别A，但实际上属于类别B，那么在错误分类后模型的梯度更新将会非常激进，导致不稳定。相比之下，蒸馏过程中的教师模型软目标促进了更稳定、连贯的梯度更新，使学生模型的学习过程更加平滑。

### 3. 在无标签数据上训练的能力
教师模型的存在允许学生模型在无标签数据上进行训练。教师模型可以为这些无标签样本生成伪标签，学生模型可以据此进行训练。这种方法显著增加了可用的训练数据量。

### 4. 视角的转变
深度学习模型通常假设给出足够的数据，它们就能近似一个函数 `F`，准确反映底层现象。然而，在许多情况下，数据稀缺使得这种假设不现实。传统的方法是构建更大的模型，并通过反复微调以获得最佳结果。相比之下，知识蒸馏则改变了这种视角：在已经拥有一个训练良好的教师模型 `F` 的前提下，目标变成使用更小的模型 `f` 来近似 `F`。