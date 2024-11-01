# 真实世界中的成像

你是否曾试过给一群小猫拍照？如果没有，那你可错过了一场美丽而混乱的景象。小猫是可爱的小生物，它们会以最疯狂的方式四处走动。它们会做出最可爱的动作，但这只会持续半秒，接着便是更可爱的动作。还没反应过来，你已经弯着腰调整镜头，试图让那只小猫出现在画面中，而此时另一只小猫正在爬你的腿。你完全沉浸在它们的毛茸茸中，根本没时间查看照片。当你坐下来查看时，照片全都模糊不清，只有一两张值得保留。你只能感叹，原以为小猫更上镜。

小猫的例子虽然简单，但它反映了在真实世界中成像为何如此困难。样本（包含小猫的场景）往往变化速度比相机调整速度更快。静止不动的相机若不追踪小猫的移动也会变得困难，因为我们的对象（小猫）在空间中移动，导致焦点变化。为了捕捉白色背景更换镜头也可能导致失真，具体取决于对象与相机的距离（见下面可爱的例子）。感兴趣的事件（小猫的可爱姿势）往往被其他成百上千张毫无趣味的照片所掩盖。小猫的例子看似无厘头，但在许多其他场景中同样存在这些困难。成像确实很难，然而，互联网上却充满了可爱的猫咪照片。

我们很容易产生这样的想法：如果我们有一台更好的相机，一台响应更快速、分辨率更高的相机，那一切问题就能迎刃而解。我们会得到我们想要的可爱照片。此外，我们将在本课程中学习如何不止是捕捉可爱的猫咪照片，还希望在监控摄像头上构建一个模型，检查小猫是否还和它们的妈妈在一起，以确保它们的安全。这听起来完美，对吧？

在我们去购买最新、最炫的相机之前，认为我们会有更好的数据、训练出一个超级准确的模型，在猫咪追踪市场上表现出色。本文将引导你朝着更有成效的方向前进，并可能为你节省大量时间和金钱。高分辨率并非所有问题的解决方案。首先，处理图像的典型神经网络模型是卷积神经网络（CNN）。CNN要求输入图像为指定大小。大图像需要更大的模型，训练时间会更长。你的电脑内存也可能有限。较大的图像尺寸意味着每次迭代训练的图像数量会减少，因为内存会被限制。

显而易见的解决方案是购买一台拥有更多GPU和内存的电脑。但这也意味着除了购买相机外，你还需要支付更多费用来训练猫咪模型。此外，这也不反映现实世界的情况。真实应用中，计算机模型通常在GPU和内存不足的环境中运行。等等，这不正是我们最初的情况吗？我们如何将模型适配到监控摄像头的硬件上？

我们有了一个想法：我们将尝试使用较小的模型来模拟大型模型的行为！顺便说一句，这确实是可行的方法。但即使这样，获取最高质量的图像也不见得是好主意，因为传输高质量图像通常需要更长的时间。即使是50Gb的猫咪图片数据，依然是50Gb的数据，无论内容多么可爱。从经济角度来看，这可能不是一个明智的选择。此外，占用整个服务器的资源也并非结交朋友的好方式。

还有一个更重要的原因不去追求最高分辨率。高分辨率不仅增强捕获目标信号的能力，也增加了拾取噪声的概率。因此，在较低分辨率图像上学习可能会更容易。较低分辨率或许可以更快地训练出更高准确度且更便宜的模型，无论是从计算成本还是资金成本上来看。这里的要点是，基于图像的噪声特性和训练与部署模型的基础设施，选择合适的分辨率。而且，既然我们要在监控摄像头上构建模型，我们不妨直接用监控摄像头拍摄的照片。

## 成像一切

成像技术令人印象深刻的一点在于我们对其不断追求。我们从不知足。这不仅适用于小猫照片，周围的世界也一样。我们天生好奇。如第一章所述，我们依赖视觉来做出决策。当决策变得困难时，我们会希望有一个清晰的视觉呈现（无意双关）。

因此，作为一个物种，我们开发出了超越肉眼捕捉范围的新方法。我们想要看到自然界最初不让我们看到的东西。我几乎可以保证，只要有我们无法确定外观的东西，总会有人尝试去成像。

人类只能看到光谱的一部分，我们称之为可见光谱。下图显示了它是多么的狭窄：

![显示可见光谱与电磁光谱的对比图 由https://open.lib.umn.edu/intropsyc/chapter/4-2-seeing/ 提供](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/human_spectrum.jpg)

要看到母自然赋予我们之外的东西，我们需要传感器在该光谱之外进行捕捉。换句话说，我们需要在不同的波长上检测信号。红外线（IR）被用于夜视设备和一些天文观测。磁共振利用强磁场和射频波成像人体软组织。我们还创造了不依赖光的观察方式。例如，电子显微镜利用电子以比传统光学更高的分辨率进行放大。超声波是另一个典型例子。超声成像利用声波生成详细的、实时的内脏器官和组织图像，提供了超越标准光学成像方法的非侵入性、动态视角。

然后，我们将巨大的镜头对准天空，用它们来勾画曾经未知的世界。我们也将目光投向微小的领域，通过成像来构建DNA结构和原子个体。这些设备都基于操控光的概念。我们利用不同类型的镜子或镜头，根据我们感兴趣的方式弯曲和聚焦光。

我们对“看见”的痴迷甚至让科学家改变了一些动物的DNA序列，以便在目标蛋白上附着一种特殊类型的蛋白（绿色荧光蛋白，GFP）。顾名思义，当绿色光波长照射样本时，GFP会发出荧光信号。这样，科学家便可以通过成像轻松地确认目标蛋白的表达位置。

接着，我们改进了这种系统，以便获得更多通道、更长时间尺度、更高分辨率。一个很好的例子是显微镜如今可以在一夜之间生成数TB的数据。

一个结合了这种努力的经典示例是下面的视频。在其中，你可以看到一条被荧光蛋白标记的鱼类胚胎发育的三维图像投影。图像中的每一个彩色小点都代表一个单独的细胞。

![鱼类胚胎图像 来源自https://www.biorxiv.org/content/10.1101/2023.03.06.531398v2.supplementary-material ](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/fish.gif)

这种成像的多样性相当非凡。这些光学工具成为我们观察宇宙的“眼睛”。它们为我们提供了改变对宇宙和生命理解的见解。我们日常使用它来分享亲人远在他乡的照片。医生在需要详细观察时会使用X光。孕妇通过超声波检查胎儿的状况。我们甚至能够拍摄到黑洞和电子这种极大与极小的事物。这听起来有点神奇，甚至带点浪漫主义色彩。

## 关于成像的视角

正如我们之前所见，我们逐渐习惯了不同的成像方式。虽然现在看起来是常规操作，但这背后花费了大量的时间和努力。我们似乎并未因此减缓探索的步伐，不断发现新的观察方式和新的成像方式。随着我们不断构建新的仪器来更好地观察，新的故事和谜团也逐渐揭开。本部分中，我们将展示一些过去已向我们揭示的谜团。

### 图片51

![图片51由Raymond Gosling/King's College London](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Photo_51_x-ray_diffraction_image.jpg)

第一张DNA图片，也被称为照片51。它采用一种基于DNA纤维晶体凝胶的纤维衍射图像技术成像。这张图片由研究生Raymond Gosling于1952年5月在Rosalind Franklin的指导下拍摄。它是1953年Watson和Crick双螺旋模型的关键证据之一。关于这张照片有许多争议，其中一部分源于Rosalind Franklin早期工作的贡献未被充分认可，以及照片在与Watson和Crick共享时的情况。然而，这张图片显著推动了我们对DNA结构的理解和随之发展的技术。

### 淡蓝色的点

![淡蓝色的点由Voyager 1拍摄](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Pale_Blue_Dot.png)

淡蓝色的点是1990年由一个空间探测器拍摄的照片。地球的大小如此之小，以至于不到一个像素。这张图片因展示了地球在广阔空间中的微小和短暂而广受关注，启发了Carl Sagan撰写《淡蓝色的点》一书。该图片由Voyager 1的1500毫米高分辨率窄角摄像机拍摄，Voyager 1也是太阳系“家庭肖像”的拍摄者。

### 黑洞

![M87由事件视界望远镜拍摄](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Black_hole_-_Messier_87.jpg)

另一个具有重要天文学意义的事件发生在2019年4月，研究人员捕获到了黑洞的首张图像！这是距离地球约5500万光年、位于室女座M87星系中心的超大质量黑洞的图像。这张引人注目的图像是由事件视界望远镜（EHT）——一个全球同步射电天文台网络——合作创建的虚拟地球大小的望远镜拍摄的。所收集的数据量非常庞大，超过了1PB，需通过物理方式进行传输处理。数据结合了近红外、X射线、毫米波长和射电观测等。此成就是事件视界望远镜协作团队多年努力的成果。

![Sag A由事件视界望远镜拍摄](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/sag-event-image.jpg)

继M87\*成功成像后，天文学家们将目标转向我们银河系中心的超大质量黑洞——人马座A\*。对人马座A\*成像具有独特的挑战，由于其较小的体积和周围环境的快速变化，其环境的变化速度远快于M87\*等较大黑洞。这种快速的运动使得捕捉到稳定、精确的结构图像变得困难，就像我们之前提到的小猫图像一样！尽管存在这些挑战，所获得的图像对于在极端引力条件下测试爱因斯坦的广义相对论具有重大意义。尽管这些观测至关重要，但它们只是用于测试广义相对论预测的众多方法之一。

### 图像，图像，图像

![从DNA解码出的视频来源 https://doi.org/10.1038/nature23017](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/horsegif_0.gif)

这是一个有趣的扭曲，它并不是一种新的成像方式，而是一种读取和存储图像的新方式。上方看到的GIF图像是存储在活细菌的DNA中的。2017年，科学家们首次实现了这一概念验证，表明生物体是一种极佳的数据存储方式。为此，他们首先将图像值转化为核苷酸代码（著名的ATCG）。然后，他们使用CRISPR系统（能够编辑DNA）将这段序列嵌入DNA中，随后对DNA重新测序并重构了所见的GIF图像。

这已经非常惊人了，但还不止于此。我们甚至可以看到实际的工作过程！虽然不是这个具体示例，但另一个科学家小组使用高速原子力显微镜展示了其工作原理。这种显微镜通过机械连接的尖端扫描表面生成样本的拓扑描述，所有这一切都是在纳米尺度下进行的。下方视频展示了CRISPR-Cas9系统——DNA编辑器——的首步过程，通过咀嚼DNA来进行编辑。美味！

![CRISPR-Cas9咀嚼DNA改编自https://doi.org/10.1038/s41467-017-01466-8](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/cas_9.gif)

不仅如此，您是否曾想过科学家如何成像DNA？不论您是否相信，这个过程同样涉及成像。要知道DNA序列，科学家首先需要制作其拷贝，这些拷贝通过为每个核苷酸（即ATCG）标记不同的荧光染料来生成。每个核苷酸会逐个匹配到序列中。添加的过程中，摄像机捕获图像。荧光颜色揭示了添加的是哪个核苷酸。通过追踪每个位置，我们可以重构DNA分子的序列。这种测序技术不仅用于重构图像，还被用于理解不同的生物过程，并在临床上有广泛应用。医生可以从这些序列中做出各种决定，例如可以对肿瘤样本进行测序，以确定其是否为侵袭性。这会生成高维数据。在如此高维的环境下得出任何结论都很困难，因此通常将其降维为2D图像。这些2D图像可以像任何图像一样进行处理，也就是说，您可以使用CNN对其进行分类。令人惊叹，对吧？

## 图像特性取决于获取方式

无论图像类型如何，所有图像都具有相同的基本特性。它们代表空间组成，通常由矩阵表示。然而，必须认识到图像并非相同创建的。图像的独特特性源于主题和获取方式。换句话说，我们不期望黑洞和DNA看起来相同，同样也不期望一个人的照片和X射线图像看起来一样。

理解图像特性是构建计算机视觉模型的良好第一步。不仅因为它会影响模型性能，还因为它将决定哪些模型更适合您的问题。显然，不是每种图像类型都需要开发新的神经网络架构。有时，您可以通过微调或更改最后一层来适应现有模型以执行不同的任务。有时不需要这种操作，而是通过预处理使图像更接近模型训练时的输入。不要担心细节，这将在本课程的后续章节中详细讲解。这里提到这些内容是为了帮助您理解成像的上下文为何重要。

对于在不同波长但同一坐标系下获取的图像，可以将每次获取视为一个不同的颜色通道。例如，通过X射线和近红外获取的图像，可以将它们视为不同的颜色通道，以此方式每个图像都有其自己的灰度图。

虽然看似简单，但某些技术（如雷达和超声波）使用一种称为极坐标网格的不同坐标系。此网格从信号发射的中心开始。与笛卡尔坐标系不同，像素大小并不一致。随着距离增加，坐标也随之增加。这意味着像素在距中心越远的区域代表的面积越大。有两种方法：第一种是将坐标系更改为像素大小一致的系统，这会导致大量信息丢失，可能不理想，且会导致次优存储。另一种方法是保持原样，但将距中心的距离作为模型的另一输入。

坐标系的影响不仅限于此。例如在卫星成像中，当多个波长在相同坐标下捕获时，可将其视为不同颜色通道，如前所述。然而，当数据处于不同坐标系下时问题会更复杂，比如结合卫星图像和

地球图像以完成任务时，坐标需要相互映射。

最后，图像获取本身存在偏差问题。我们可以宽松地将偏差定义为数据集中的非期望特征，无论是噪声还是对模型行为的影响。偏差来源多种多样，但成像中的一个相关偏差是测量偏差。测量偏差出现在训练模型的数据集与实际看到的数据集差异过大时，就像我们前面提到的高清小猫图片和监视摄像头的例子。其他偏差源包括标签标记者的测量偏差（即不同群体和个体标记图像的方式不同），或图像的上下文（例如，在区分猫和狗的过程中，如果所有猫的照片都在沙发上，模型可能学会区分沙发与非沙发，而不是猫和狗）。

综上所述，识别并解决不同仪器生成的图像特性是构建计算机视觉模型的良好第一步。在这种情况下，通过预处理技术和策略来解决我们识别的问题，以减轻其对模型的影响。《计算机视觉任务的预处理》章节将深入探讨用于提升模型性能的具体预处理方法。
