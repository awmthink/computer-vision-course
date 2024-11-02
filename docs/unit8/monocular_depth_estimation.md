# 度量和相对单目深度估计：概述。微调 Depth Anything V2 👐 📚

## 模型的演变

在过去的十年中，单目深度估计模型取得了显著的进步。让我们通过视觉之旅来回顾这一演变。

我们从这样的基本模型开始：

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/depth_estimation_evolution1.png)
发展到更复杂的模型：

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/depth_estimation_evolution2.png)

而现在，我们有了最先进的模型，Depth Anything V2：

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/depth_estimation_evolution3.png)

很令人惊叹，不是吗？

今天，我们将揭开这些模型的工作原理，简化复杂的概念。此外，我们将使用自定义数据集微调我们自己的模型。“但是等等，”你可能会问，“当最新的模型在任何环境中都表现得如此出色时，为什么我们需要在自己的数据集上微调模型呢？”

这就是细微差别和具体情况发挥作用的地方，而这正是本文的重点。如果你渴望探索单目深度估计的复杂性，请继续阅读。

## 基础

“好的，深度到底是什么？”通常，它是一个单通道图像，其中每个像素代表从相机或传感器到与该像素对应的空间点的距离。然而，事实证明，这些距离可以是绝对的或相对的——真是个转折！
- **绝对深度**：每个像素值直接对应一个物理距离（例如，以米或厘米为单位）。
- **相对深度**：像素值指示哪些点更近或更远，而不参考现实世界的测量单位。相对深度通常是反转的，即数字越小，点越远。

我们稍后将更详细地探讨这些概念。

“那么，单目是什么意思？”它只是意味着我们需要仅使用一张照片来估计深度。这有什么挑战呢？看看这个：

![image/gif](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/depth_ambiguity1.gif)

![image/gif](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/depth_ambiguity2.gif)

如你所见，由于透视，将 3D 空间投影到 2D 平面上会产生歧义。为了解决这个问题，有使用多张图像进行深度估计的精确数学方法，例如立体视觉、运动结构和更广泛的摄影测量领域。此外，可以使用激光扫描仪（例如 LiDAR）等技术进行深度测量。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/stereo_vision_sfm.png)

## 相对深度和绝对深度（又名度量深度）估计：有什么意义？

让我们探讨一些突出相对深度估计必要性的挑战。为了更科学，让我们参考一些论文。

>预测度量深度的优势在于它在计算机视觉和机器人技术中的许多下游应用中具有实际用途，例如映射、规划、导航、物体识别、3D 重建和图像编辑。然而，在多个数据集上训练单个度量深度估计模型通常会降低性能，特别是当收集的图像在深度尺度上有很大差异时，例如室内和室外图像。因此，当前的 MDE 模型通常过度拟合特定数据集，并且不能很好地推广到其他数据集。

通常，这种图像到图像任务的架构是一个编码器 - 解码器模型，如 U-Net，并进行了各种修改。形式上，这是一个逐像素回归问题。想象一下，对于神经网络来说，准确预测每个像素的距离是多么具有挑战性，距离范围从几米到几百米。<br>这让我们想到放弃在所有场景中预测精确距离的通用模型。相反，让我们开发一个近似（相对）预测深度的模型，通过指示哪些物体相对于彼此和我们更远，哪些更近，来捕捉场景的形状和结构。如果需要精确距离，我们可以在特定数据集上微调这个相对模型，利用它对任务的现有理解。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/different_scales.png)

还有更多细节我们需要注意。

>模型不仅必须处理使用不同相机和相机设置拍摄的图像，还必须学会调整场景整体尺度的巨大变化。

除了不同的尺度，如我们前面提到的，一个重大问题在于相机本身，它们可以对世界有截然不同的视角。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/focal_lenght.png)

注意焦距的变化如何极大地改变对背景距离的感知！

最后，许多数据集完全没有绝对深度图，只有相对深度图（例如，由于缺乏相机校准）。此外，每种获取深度的方法都有其自身的优点、缺点、偏差和问题。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/metrics1.png)

>我们确定了三个主要挑战。1）深度的固有不同表示：直接深度表示与反向深度表示。2）尺度模糊性：对于某些数据源，深度仅在未知尺度下给出。3）位移模糊性：一些数据集仅在未知尺度和全局视差位移下提供视差，全局视差位移是未知基线和由于后处理引起的主点水平位移的函数。

*视差是指从两个不同视点观察物体时物体的明显位置差异，常用于立体视觉中估计深度。*

简而言之，我希望我已经说服你，你不能只是从互联网上获取分散的深度图，并使用一些逐像素的均方误差来训练模型。

但是我们如何平衡所有这些变化呢？我们如何尽可能地从差异中抽象出来，并从所有这些数据集中提取共性——即场景的形状和结构、物体之间的比例关系，指示什么更近，什么更远？

## 尺度和位移不变损失😎

简而言之，我们需要对要进行训练的所有深度图进行某种归一化处理，并使用该归一化后的深度图评估指标。这里有一个想法：我们希望创建一个不考虑环境尺度或各种位移的损失函数。剩下的任务是将这个想法转化为数学术语。

>具体来说，首先通过$d=\frac{1}{t}$将深度值转换到视差空间，然后在每个深度图上将其归一化到$0\sim1$。为了实现多数据集联合训练，我们采用仿射不变损失来忽略每个样本的未知尺度和位移：
$$\mathcal{L}_1=\frac{1}{HW}\sum_{i=1}^{HW}\rho(d_i^*,d_i),$$
其中$d_i^*$和$d_i$分别是预测值和真实值。而$\rho$是仿射不变平均绝对误差损失：$\rho(d_i^*,d_i)=|\hat{d}_i^*-\hat{d}_i|$，其中$\hat{d}_i^*$和$\hat{d}_i$是经过缩放和位移后的预测值$d_i^*$和真实值$d_i$：
$$\hat{d}_i=\frac{d_i-t(d)}{s(d)},$$
其中$t(d)$和$s(d)$用于使预测值和真实值具有零平移和单位尺度：
$$t(d)=\mathrm{median}(d),\quad s(d)=\frac{1}{HW}\sum_{i=1}^{HW}|d_i-t(d)|.$$

实际上，还有许多其他方法和函数有助于消除尺度和位移。损失函数也有不同的添加项，例如梯度损失，它不关注像素值本身，而是关注它们变化的速度（因此得名——梯度）。你可以在[MiDaS](https://arxiv.org/pdf/1907.01341)论文中了解更多相关内容，我将在最后列出一些有用的文献。在进入最激动人心的部分——使用自定义数据集对绝对深度进行微调之前，让我们先简要讨论一下指标。

## 指标

在深度估计中，有几个标准指标用于评估性能，包括平均绝对误差（MAE）、均方根误差（RMSE）以及它们的对数变化形式，以平滑距离中的大差距。此外，考虑以下内容：
- **绝对相对误差（AbsRel）**：这个指标与 MAE 类似，但以百分比表示，测量预测距离与真实距离平均相差的百分比。<br>$\text{AbsRel}=\frac{1}{N}\sum_{i=1}^{N}\frac{|d_i-\hat{d}_i|}{d_i}$
- **阈值准确率（$\delta_1$）**：这个指标测量预测像素与真实像素相差不超过 25%的比例。<br>$\delta_1=\text{预测深度中满足}\max\left(\frac{d_i}{\hat{d}_i},\frac{\hat{d}_i}{d_i}\right)<1.25\text{的比例}$

### 重要考虑因素
>对于我们所有的模型和基线，在测量误差之前，我们会对每张图像的预测值和真实值进行尺度和位移对齐。

实际上，如果我们正在训练以预测相对深度，但想在具有绝对值的数据集上测量质量，并且我们对在这个数据集上进行微调或对绝对值不感兴趣，我们可以像损失函数一样，从计算中排除尺度和位移，并将所有内容标准化为统一的度量。

### 计算指标的四种方法

理解这些方法有助于在分析论文中的指标时避免混淆：

1. **零样本相对深度估计**
    - 在一组数据集上训练以预测相对深度，并在其他数据集上测量质量。由于深度是相对的，显著不同的尺度不是问题，其他数据集上的指标通常仍然很高，类似于训练数据集的测试集。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/metrics2.png)

2. **零样本绝对深度估计**
    - 训练一个通用的相对模型，然后在一个好的数据集上对其进行微调以预测绝对深度，并在不同的数据集上测量绝对深度预测的质量。在这种情况下，指标往往比前一种方法更差，突出了在不同环境中很好地预测绝对深度的挑战。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/metrics3.png)

3. **微调（域内）绝对深度估计**
    - 与前一种方法类似，但现在在用于微调绝对深度预测的数据集的测试集上测量质量。这是最实用的方法之一。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/metrics4.png)

4. **微调（域内）相对深度估计**
    - 训练以预测相对深度，并在训练数据集的测试集上测量质量。这可能不是最准确的名称，但想法很简单。

## Depth Anything V2 绝对深度估计微调

在本节中，我们将通过在 NYU-D 数据集上微调模型以预测绝对深度来重现 Depth Anything V2 论文中的结果，旨在实现与上一节最后一个表格中所示的指标类似的指标。

### Depth Anything V2 背后的关键思想
Depth Anything V2 是一个强大的深度估计模型，由于几个创新概念而取得了显著的结果：

- **异构数据上的通用训练方法**：这个方法在 MiDaS 2020 论文中引入，能够在各种类型的数据集上进行稳健的训练。
- **DPT 架构**：“用于密集预测的视觉 Transformer”论文提出了这种架构，它本质上是一个带有视觉 Transformer（ViT）编码器的 U-Net 以及一些修改。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/dpt.png)

- **DINOv2 编码器**：这个标准的 ViT 通过在大规模数据集上使用自监督方法进行预训练，作为一个强大而通用的特征提取器。近年来，计算机视觉研究人员一直致力于创建类似于自然语言处理中的 GPT 和 BERT 的基础模型，DINOv2 是朝着这个方向迈出的重要一步。
- **使用合成数据**：训练流程在下面的图像中得到了很好的描述。这种方法使作者能够在深度图中实现如此清晰和准确的效果。毕竟，如果你仔细想想，从合成数据中获得的标签确实是“真实值”。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/DA2_pipeline.png)

### 微调入门

现在，让我们深入代码。如果你无法使用强大的 GPU，我强烈推荐使用 Kaggle 而不是 Colab。Kaggle 有几个优点：
- 每周最多 30 小时的 GPU 使用时间。
- 没有连接中断。
- 非常快速且方便地访问数据集。
- 在其中一种配置中能够同时使用两个 GPU，这将帮助你练习分布式训练。

你可以使用这个[Kaggle 笔记本](https://www.kaggle.com/code/amanattheedge/depth-anything-v2-metric-fine-tunning-on-nyu/notebook)直接进入代码。

我们将在这里详细介绍所有内容。首先，让我们从作者的存储库下载所有必要的模块以及带有 ViT-S 编码器的最小模型的检查点。

#### 步骤 1：克隆存储库并下载预训练权重

```bash
!git clone https://github.com/DepthAnything/Depth-Anything-V2
!wget -O depth_anything_v2_vits.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true
```
你也可以[在这里](http://datasets.lids.mit.edu/fastdepth/data/)下载数据集。

#### 步骤 2：导入所需模块

```python
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import random
import h5py

import sys
sys.path.append('/kaggle/working/Depth-Anything-V2/metric_depth')

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import notebook_launcher
from accelerate import DistributedDataParallelKwargs

import transformers

import torch
import torchvision
from torchvision.transforms import v2
from torchvision.transforms import Compose
import torch.nn.functional as F
import albumentations as A

from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
```

#### 步骤 3：获取训练和验证的所有文件路径

```python
def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


train_paths = get_all_files('/kaggle/input/nyu-depth-dataset-v2/nyudepthv2/train')
val_paths = get_all_files('/kaggle/input/nyu-depth-dataset-v2/nyudepthv2/val')
```

#### 步骤 4：定义 PyTorch 数据集

```python
#NYU Depth V2 40k. Original NYU is 400k
class NYU(torch.utils.data.Dataset):
    def __init__(self, paths, mode, size=(518, 518)):
        
        self.mode = mode #train or val
        self.size = size
        self.paths = paths
        
        net_w, net_h = size
        #作者的变换
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
        
        # 仅在论文中有水平翻转
        self.augs = A.Compose([
            A.HorizontalFlip(),
            A.ColorJitter(hue = 0.1, contrast=0.1, brightness=0.1, saturation=0.1),
            A.GaussNoise(var_limit=25),
        ])
    
    def __getitem__(self, item):
        path = self.paths[item]
        image, depth = self.h5_loader(path)
        
        if self.mode == 'train':
            augmented = self.augs(image=image, mask = depth)
            image = augmented["image"] / 255.0
            depth = augmented['mask']
        else:
            image = image / 255.0
          
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        # 有时数据集中由于噪声等原因，深度图中会有有效的深度掩码。
#         sample['valid_mask'] =... 
     
        return sample

    def __len__(self):
        return len(self.paths)
    
    def h5_loader(self, path):
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth'])
        return rgb, depth
```

这里有几点需要注意：
- 原始的 NYU-D 数据集包含 40.7 万个样本，但我们使用的是 4 万个样本的子集。这将略微影响最终模型的质量。
- 论文的作者仅使用水平翻转进行数据增强。
- 偶尔，深度图中的某些点可能无法正确处理，导致“坏像素”。一些数据集除了图像和深度图之外，还包括一个掩码，用于区分有效和无效像素。这个掩码对于在损失和指标计算中排除坏像素是必要的。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/depth_holes.png)

- 在训练期间，我们调整图像大小，使较小的边为 518 像素，然后进行裁剪。对于验证，我们不裁剪或调整深度图的大小。相反，我们对预测的深度图进行上采样，并在原始分辨率下计算指标。

#### 步骤 5：数据可视化

```python
num_images = 5

fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

train_set = NYU(train_paths, mode='train') 

for i in range(num_images):
    sample = train_set[i*1000]
    img, depth = sample['image'].numpy(), sample['depth'].numpy()

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = img*std+mean

    axes[i, 0].imshow(np.transpose(img, (1,2,0)))
    axes[i, 0].set_title('图像')
    axes[i, 0].axis('off')

    im1 = axes[i, 1].imshow(depth, cmap='viridis', vmin=0)
    axes[i, 1].set_title('真实深度')
    axes[i, 1].axis('off')
    fig.colorbar(im1, ax=axes[i, 1])
    
plt.tight_layout()

```
![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/dataset.png)

如你所见，图像非常模糊且有噪声。因此，我们无法获得在 Depth Anything V2 预览中看到的细粒度深度图。在黑洞伪影中，深度为 0，我们稍后将利用这一事实来掩盖这些空洞。此外，数据集中包含许多同一位置几乎相同的照片。

#### 步骤 6：准备数据加载器

```python
def get_dataloaders(batch_size):
    
    train_dataset = NYU(train_paths, mode='train')
    val_dataset = NYU(val_paths, mode='val')
    
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle=True,
                                                  num_workers=4,
                                                  drop_last=True
                                                  )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                               batch_size = 1, #用于无填充的动态分辨率评估
                                               shuffle=False,
                                               num_workers=4,
                                               drop_last=True
                                                )
    
    return train_dataloader, val_dataloader

```
#### 步骤 7：指标评估

```python
def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    mae = torch.mean(torch.abs(diff))

    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.detach(), 'abs_rel': abs_rel.detach(),'rmse': rmse.detach(), 'mae': mae.detach(), 'silog':silog.detach()}

```
我们的损失函数是 SiLog。似乎在对绝对深度进行训练时，我们应该忘记尺度不变性和其他用于相对深度训练的技术。然而，事实证明这并不完全正确，我们通常仍然希望使用一种“尺度正则化”，但程度较轻。参数λ=0.5 有助于在全局一致性和局部准确性之间取得平衡。

#### 步骤 8：定义超参数

```python
model_weights_path =  '/kaggle/working/depth_anything_v2_vits.pth' 
model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
model_encoder = 'vits'
max_depth = 10
batch_size = 11
lr = 5e-6
weight_decay = 0.01
num_epochs = 10
warmup_epochs = 0.5
scheduler_rate = 1
load_state = False

state_path = "/kaggle/working/cp"
save_model_path = '/kaggle/working/model'
seed = 42
mixed_precision = 'fp16'
```
注意参数“**最大深度**”。我们模型中的最后一层是对每个像素的 sigmoid 函数，产生 0 到 1 的输出。我们只需将每个像素乘以“**最大深度**”，以表示从 0 到“**最大深度**”的距离。

#### 步骤 9：训练函数

```python
def train_fn():

    set_seed(seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) 
    accelerator = Accelerator(mixed_precision=mixed_precision, 
                              kwargs_handlers=[ddp_kwargs],
                             )
    
    # 在论文中，他们随机初始化解码器并仅使用编码器预训练权重。然后进行全模型微调
    # ViT-S 编码器在这里
    model = DepthAnythingV2(**{**model_configs[model_encoder], 'max_depth': max_depth})
    model.load_state_dict({k: v for k, v in torch.load(model_weights_path).items() if 'pretrained' in k}, strict=False)
    
    optim = torch.optim.AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': lr*10}],
                      lr=lr, weight_decay=weight_decay)
    
    criterion = SiLogLoss() # 作者的损失函数
    
    train_dataloader, val_dataloader = get_dataloaders(batch_size)
    
    scheduler = transformers.get_cosine_schedule_with_warmup(optim, len(train_dataloader)*warmup_epochs, num_epochs*scheduler_rate*len(train_dataloader))
    
    model, optim, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optim, train_dataloader, val_dataloader, scheduler)
    
    if load_state:
        accelerator.wait_for_everyone()
        accelerator.load_state(state_path)
        
    best_val_absrel = 1000
    
    
    for epoch in range(1, num_epochs):
        
        model.train()
        train_loss = 0
        for sample in tqdm(train_dataloader, disable = not accelerator.is_local_main_process):
            optim.zero_grad()
            
            img, depth = sample['image'], sample['depth']
            
            pred = model(img) 
                                                     # 掩码
            loss = criterion(pred, depth, (depth <= max_depth) & (depth >= 0.001))
            
            accelerator.backward(loss)
            optim.step()
            scheduler.step()
            
            train_loss += loss.detach()
            
            
        train_loss /= len(train_dataloader)
        train_loss = accelerator.reduce(train_loss, reduction='mean').item()
        
        
        model.eval()
        results = {'d1': 0, 'abs_rel': 0,'rmse': 0, 'mae': 0, 'silog': 0}
        for sample in tqdm(val_dataloader, disable = not accelerator.is_local_main_process):
            
            img, depth = sample['image'].float(), sample['depth'][0]
            
            with torch.no_grad():
                pred = model(img)
                # 在原始分辨率下评估
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            valid_mask = (depth <= max_depth) & (depth >= 0.001)
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            

        for k in results.keys():
            results[k] = results[k] / len(val_dataloader)
            results[k] = round(accelerator.reduce(results[k], reduction='mean').item(),3)
        
        accelerator.wait_for_everyone()
        accelerator.save_state(state_path, safe_serialization=False)
        
        if results['abs_rel'] < best_val_absrel:
            best_val_absrel = results['abs_rel']
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_local_main_process:
                torch.save(unwrapped_model.state_dict(), save_model_path)
        
        accelerator.print(f"epoch_{epoch},  train_loss = {train_loss:.5f}, val_metrics = {results}")
        
# 注意：在测试一种配置时，我遇到了损失变为 nan 的错误。
# 通过在预测中添加一个小的epsilon 来防止除以 0 可以修复此问题
```
在论文中，作者随机初始化解码器并仅使用编码器权重。然后他们对整个模型进行微调。其他值得注意的点包括：
- 对解码器和编码器使用不同的学习率。编码器的学习率较低，因为我们不想像对随机初始化的解码器那样显著改变已经很好的权重。
- 作者在论文中使用了多项式调度器，而我使用了带有预热的余弦调度器，因为我喜欢它。
- 在掩码中，如前所述，我们通过使用条件“**depth >= 0.001**”来避免深度图中的黑洞。
- 在训练周期中，我们在调整大小后的深度图上计算损失。在验证期间，我们对预测进行上采样并在原始分辨率下计算指标。
- 看看我们可以多么容易地使用 HF accelerate 为分布式计算包装自定义 PyTorch 代码。

#### 步骤 10：启动训练

```python
# 你可以使用 1 个 GPU 运行此代码。只需将 num_processes=1
notebook_launcher(train_fn, num_processes=2)
```

我相信我们已经实现了预期的目标。性能上的微小差异可归因于数据集大小的显著差异（40k 与 400k）。请记住，我们使用了 ViT-S 编码器。

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/metrics5.png)

让我们展示一些结果

```python
model = DepthAnythingV2(**{**model_configs[model_encoder], 'max_depth': max_depth}).to('cuda')
model.load_state_dict(torch.load(save_model_path))

num_images = 10

fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

val_dataset = NYU(val_paths, mode='val') 
model.eval()
for i in range(num_images):
    sample = val_dataset[i]
    img, depth = sample['image'], sample['depth']
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
   
    with torch.inference_mode():
        pred = model(img.unsqueeze(0).to('cuda'))
        pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
    img = img*std + mean
     
    axes[i, 0].imshow(img.permute(1,2,0))
    axes[i, 0].set_title('图像')
    axes[i, 0].axis('off')

    max_depth = max(depth.max(), pred.cpu().max())
    
    im1 = axes[i, 1].imshow(depth, cmap='viridis', vmin=0, vmax=max_depth)
    axes[i, 1].set_title('真实深度')
    axes[i, 1].axis('off')
    fig.colorbar(im1, ax=axes[i, 1])
    
    im2 = axes[i, 2].imshow(pred.cpu(), cmap='viridis', vmin=0, vmax=max_depth)
    axes[i, 2].set_title('预测深度')
    axes[i, 2].axis('off')
    fig.colorbar(im2, ax=axes[i, 2])

plt.tight_layout()
```

![image/png](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Metric%20and%20Relative%20Monocular%20Depth%20Estimation%20An%20Overview.%20Fine-Tuning%20Depth%20Anything%20V2/inference.png)

验证集中的图像比训练集中的图像更清晰、更准确，这就是为什么我们的预测相比之下显得有点模糊的原因。再看看上面的训练样本。

总的来说，关键要点是模型的质量在很大程度上取决于提供的深度图的质量。Depth Anything V2 的作者克服了这一限制并生成了非常清晰的深度图，值得称赞。唯一的缺点是它们是相对深度。

## 参考文献
- [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/pdf/1907.01341)
- [ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/pdf/2302.12288)
- [Vision Transformers for Dense Prediction](https://arxiv.org/pdf/2103.13413)
- [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/pdf/2401.10891)
- [Depth Anything V2](https://arxiv.org/pdf/2406.09414)