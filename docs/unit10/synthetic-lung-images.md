# 使用DCGAN生成合成数据

我们在单元5中学习了GAN是一种机器学习框架，包含两个神经网络：生成器（Generator）和判别器（Discriminator）。生成器用于创建合成图像，而判别器则试图区分真实图像和虚假图像。通过这种对抗性的过程，生成器逐步提升生成逼真图像的能力，而判别器则不断提高区分真实和虚假图像的能力。

我们将研究如何利用GAN生成医学图像。医学图像领域面临小数据集、隐私问题以及有限的标注样本量的挑战。研究人员已经使用GAN生成了肺部X光图像、视网膜图像、脑扫描图像和肝脏图像等合成图像。在[基于GAN的合成脑PET图像生成](https://braininformatics.springeropen.com/counter/pdf/10.1186/s40708-020-00104-2.pdf)中，作者生成了阿尔茨海默病三个不同阶段的脑PET图像。另一项研究[基于GAN的合成医学图像增强以提高肝脏病变分类中CNN的表现](https://arxiv.org/abs/1803.01229)生成了合成肝脏图像。[BrainGAN: 利用GAN架构和CNN模型的脑MRI图像生成与分类框架](https://www.mdpi.com/1424-8220/22/11/4297)开发了用于生成脑MRI图像的框架，[一种基于DCGAN和深度迁移学习的COVID-19检测模型](https://www.sciencedirect.com/science/article/pii/S1877050922007463)则利用DCGAN生成了合成肺部X光图像以协助COVID-19检测。

## DCGAN（深度卷积生成对抗网络）

DCGAN是在[基于深度卷积生成对抗网络的无监督表示学习](https://arxiv.org/abs/1511.06434)中由Radford等人提出的模型，许多研究人员使用它生成医学合成图像。我们将使用它来生成合成肺部图像。在使用DCGAN进行模型训练之前，我们将简要回顾其架构。生成器网络以随机噪声为输入，生成合成肺部图像，而判别器网络则尝试区分真实图像和合成图像。DCGAN在生成器和判别器中都使用卷积层来有效地捕获空间特征，同时用步幅卷积替换了最大池化进行空间维度的下采样。

生成器的模型架构如下：

- 输入是包含100个随机数的向量，输出是尺寸为128\*128\*3的图像。
- 模型包含4个卷积层：
  - Conv2D层
  - 批量归一化层
  - ReLU激活
- Conv2D层带有Tanh激活。

判别器的模型架构如下：

- 输入是一张图像，输出是一个概率，指示图像是虚假图像还是真实图像。
- 模型包含一个卷积层：
  - Conv2D层
  - Leaky ReLU激活
- 三个卷积层，包含：
  - Conv2D层
  - 批量归一化层
  - Leaky ReLU激活
- Conv2D层带有Sigmoid激活。

**数据收集**

首先，我们需要获取一个[数据集](https://data.mendeley.com/datasets/rscbjbr9sj/2)的真实肺部图像。我们将从Hugging Face Hub下载[Chest X-Ray Images (Pneumonia)](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)数据集。

以下是有关数据集的信息：

    * 出自[基于图像的深度学习识别医学诊断和可治疗疾病](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867418301545%3Fshowall%3Dtrue):

    * 数据集组织为3个文件夹（train、test、val），包含每个图像类别（Pneumonia/Normal）的子文件夹。共有5863张X光图像（JPEG）和2个类别（Pneumonia/Normal）。

    * 胸部X光图像（前后视）选自广州市妇女儿童医疗中心1至5岁儿童的回顾性队列，所有胸部X光影像均为患者例行临床护理的一部分。

    * 为了分析胸部X光图像，所有胸片最初进行了质量控制筛选，去除低质量或不可读的扫描图像。图像的诊断结果由两名专业医生评级，经过审核后用于AI系统的训练。为了防止评级误差，评估集还由第三位专家审核。

我们将首先登录Hugging Face Hub。

```python
from huggingface_hub import notebook_login

notebook_login()
```

接下来，加载数据集。

```python
from datasets import load_dataset

dataset = load_dataset("hf-vision/chest-xray-pneumonia")
```

我们将通过调整大小和标准化像素值来预处理肺部图像。

```python
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

transform = Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
)
```

在训练过程中，生成器的目标是生成难以区分的合成肺部图像，而判别器则学习正确分类图像为真实或合成。我们从初始化生成器随机噪声开始，进行100轮训练。

让我们可视化训练过程：

![trainig-gif](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/dcgan_training_animation.gif)

## 我们的成果如何？

以下是64张“好的”合成图像，定义为获得判别器70%概率的“真实”标签。

![lung-images](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/good_images.png)

可以看到一些合成肺部图像效果不错，但另一些显得模糊。有几点重要的注意事项。首先，生成合成医学图像的研究人员通常会引入“中间人”--在此情况下为放射科专家--来评估合成图像。只有能够骗过专家的图像才会与真实数据一起用于模型训练。其次，那些看起来不错的生成图像通常看起来非常相似。这是GAN的一个已知问题--“模式崩溃”。简单来说，当生成器开始重复输出同一类型的图像时，这种现象就会发生。可以类比为某人得到了很多关于制作巧克力曲奇的赞扬，因此变得_非常非常_擅长制作这种曲奇，但无法制作其他类型的曲奇。

鉴于使用GAN生成高质量医学图像的已知挑战，一些研究人员探索了使用扩散模型生成肺部图像的方法。Medfusion是一种用于医学图像的条件潜在DDPM模型，提出于[扩散概率模型在医学2D图像上击败GAN](https://arxiv.org/pdf/2212.07501.pdf)。在[合成数据在医学影像研究中的潜力](https://arxiv.org/pdf/2311.09402.pdf)中，Khosravi等人发现使用一种结合真实图像和合成图像的扩散过程提高了模型性能。

## 资源和延伸阅读

- [基于DCGAN和深度迁移学习的新型COVID-19检测模型](https://www.sciencedirect.com/science/article/pii/S1877050922007463)
- [Augmentation_Gan](https://github.com/rossettisimone/AUGMENTATION_GAN)
- [BrainGAN: 利用GAN架构和CNN模型的脑MRI图像生成与分类框架](https://www.mdpi.com/1424-8220/22/11/4297)
- [基于图像的深度学习识别医学诊断和可治疗疾病](https://www.cell.com/action/showPdf?pii=S0092-8674%2818%2930154-5)
- [扩散概率模型在医学图像上击败GAN](https://arxiv.org/abs/2212.07501)
- [DR-DCGAN：用于糖尿病视网膜病变图像合成的深度卷积生成对抗网络](<https://www.webology.org/data-cms/articles/20220204053948pmwebology%2019%20(2)%20-%2077%20.pdf>)
- [改进脑肿瘤分割的Deepfake图像生成](https://aps.arxiv.org/abs/2307.14273)
- [GAN实验室](https://pol

oclub.github.io/ganlab/)
- [用于医学图像合成的GAN：实证研究](https://arxiv.org/abs/2105.05318)
- [Medfusion Github repo](https://github.com/mueller-franzes/medfusion)
- [生成对抗网络潜在空间中的医学图像编辑](https://www.sciencedirect.com/science/article/pii/S2666521221000168?ref=pdf_download&fr=RR-2&rr=833e48fa5e777142)
- [带有上下文感知的GAN用于医学图像合成](https://arxiv.org/abs/1612.05362)
- [MedSynAnalyzer](https://github.com/ayanglab/MedSynAnalyzer)[dcgan_faces_tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py)
- [StudioGAN：图像合成的GAN分类和基准](https://arxiv.org/abs/2206.09479)
- [PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)
- [基于深度卷积生成对抗网络的无监督表示学习](https://arxiv.org/abs/1511.06434)