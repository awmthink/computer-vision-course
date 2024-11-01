# 什么是计算机视觉

让我们回顾上一个章节的示例：踢球。正如我们所见，这涉及到大脑能够瞬间完成的多个任务。从图像输入中提取有意义的信息是计算机视觉的核心。那么，什么是计算机视觉呢？

## 定义

计算机视觉是让机器具备“看”能力的科学和技术。它涉及开发理论和算法方法，以获取、处理、分析和理解视觉数据，并使用这些信息生成对世界的有意义的表示、描述和解释（*Forsyth & Ponce, Computer Vision: A Modern Approach*）。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/743a2a115b53f258c9e6bc7744534d9e03b8a124/CV_in_defintiion.png" alt="计算机视觉与其他领域的关系">
</div>

## 深度学习与计算机视觉的复兴

计算机视觉的演变标志着跨学科领域的一系列渐进式发展，每一步都带来了突破性的算法、硬件和数据，为其赋予了更强大的能力和灵活性。其中一个重要的飞跃是深度学习方法的广泛应用。

最初，为了在图像中提取和学习信息，我们通过图像预处理技术提取特征。当获得了一组描述图像的特征后，再在特征数据集上应用经典的机器学习算法。这种方法简化了从硬编码规则中提取特征的过程，但仍然依赖于领域知识和详尽的特征工程。而当深度学习（DL）方法与大型数据集相结合时，出现了一种更先进的方法。深度学习允许机器从原始数据中自动学习复杂特征。这种范式转变使我们能够构建更加自适应和复杂的模型，促成了计算机视觉领域的复兴。

计算机视觉的种子早在1960年代深度学习模型兴起之前就已播下，像David Marr和Hans Moravec这样的先驱者曾思考过一个根本性问题：我们能让机器“看见”吗？早期的突破，如边缘检测算法、物体识别，借助巧妙的设计与强力计算相结合，为构建计算机视觉系统奠定了基础。随着研究开发的进步和硬件能力的提升，计算机视觉社区迅速扩展，汇聚了来自全球不同学科的研究人员、工程师、数据科学家以及充满热情的业余爱好者。通过开源和社区驱动的项目，我们正在见证获取前沿工具和技术的民主化，这些项目帮助推动了该领域的复兴。

## 与其他领域的跨学科性和图像理解

就像很难明确区分人工智能和计算机视觉一样，也难以将计算机视觉与其邻近领域完全分开。例如图像预处理和分析的案例。一个尝试性的区分是，图像分析的输入和输出始终是图像。然而，这是一种短视的看法。即使是简单的任务，比如计算图像的中值，也可以归入计算机视觉范畴。为澄清它们的差异，我们必须引入一个新概念：图像理解。

图像理解是理解图像内容的过程，可以分为三个不同的层次：

**低层次处理** 是对图像的基本操作（如图像锐化、对比度调整）。输入和输出都是图像。

**中层次处理** 包括分割、物体描述和物体分类。输入是图像，而结果是与图像关联的属性。这可以通过图像预处理和机器学习算法的结合完成。

**高层次处理** 则涉及对整个图像进行理解，例如识别给定的物体、场景重建和图像转文字。通常这些任务与人类认知相关。

图像分析主要关注低层次和中层次处理，而计算机视觉则更关注中层次和高层次处理。因此，在图像分析和计算机视觉之间，中层次处理存在重叠。

这是一个需要牢记的点，因为在数据量少或图像简单的场景中，分配资源来开发复杂模型（如神经网络）可能并不合适。从业务角度来看，模型开发需要时间和资金，因此了解何时使用合适的工具是必要的。

在进入更强大的模型之前，通常会先进行“预处理”部分。相反，有时神经网络的层次会自动执行这些任务，省去了显式预处理的需求。对于熟悉数据科学的人来说，图像分析可能是数据探索性分析的第一步。最后，经典图像分析方法也可以用于数据增强，以提高计算机视觉模型的训练数据的质量和多样性。

## 计算机视觉任务概览

我们已经看到，对于计算机而言，计算机视觉非常困难，因为它们没有世界的先验知识。在我们的例子中，我们知道球是什么，如何跟踪其运动，物体在空间中通常如何移动，如何估算球何时会到达我们身边，脚的位置、脚的运动方式以及估计踢球所需的力度。如果我们将其分解为具体的计算机视觉任务，我们将会得到以下内容：
  - 场景识别
  - 物体识别
  - 物体检测
  - 分割（实例分割、语义分割）
  - 跟踪
  - 动态环境适应
  - 路径规划

在《计算机视觉任务》一章中，您将进一步了解计算机视觉的核心任务。但计算机视觉能做的任务远不止于此！以下是一个不完整的任务清单：
  - 图像标注
  - 图像分类
  - 图像描述
  - 异常检测
  - 图像生成
  - 图像恢复
  - 自主探索
  - 定位

## 任务复杂性

图像分析和计算机视觉领域中任务的复杂性并不单纯取决于任务的崇高性或难度。相反，它主要取决于被分析的图像或数据的属性。以识别图像中的行人为例。对人类观察者而言，这似乎是一个简单的任务，因为我们擅长识别人类。然而，从计算的角度来看，任务的复杂性可能会因照明条件、遮挡情况、图像分辨率和摄像头质量等因素而显著变化。在光线较暗或像素化的图像中，即使是简单的行人检测任务对计算机视觉算法来说也会变得非常复杂，需要使用先进的图像增强和机器学习技术。因此，图像分析和计算机视觉的挑战往往不在于任务的崇高性，而在于视觉数据的复杂性和提取有意义见解所需的计算方法的复杂性。

## 计算机视觉应用链接

作为一个领域，计算机视觉在社会中的重要性日益增加。关于其应用有许多伦理考虑。例如，如果一个用于检测癌症的模型将癌症样本误分类为健康样本，可能会产生严重后果。能够跟踪人的监控技术同样引发了许多隐私问题。这将在“单元12——伦理与偏见”中详细讨论。在“计算机视觉的应用”中，我们将让您体验一些很酷的应用。