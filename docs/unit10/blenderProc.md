# 使用3D渲染器生成合成数据

在创建计算机生成的图像作为合成训练数据时，我们理想地希望这些图像看起来尽可能真实。物理渲染器（PBR）如[Blender Cycles](https://www.blender.org)或[Unity](https://unity.com)能够帮助创建超真实的图像，使其在外观和感觉上都像现实世界。

假设你在创建一个闪亮的苹果图像。当你为这个苹果着色时，你希望它看起来逼真，对吗？这就是PBR的用武之地。

![apple](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/apple.jpg)

好，我们来详细讲解一下：

_颜色和光线：_

- 当光线照射到物体上时，它与物体的互动方式各不相同。PBR试图模拟这种互动。
- 想象一下光线照射到苹果上的情景。一些部分会因为光线直接照射而更亮，而其他部分因为光线被遮挡或不够强烈而显得更暗。

_材质：_

- 不同的材质对光线的反应不同。例如，光滑的金属表面反射光线多于柔软的哑光织物。
- PBR会考虑物体的材质，因此，如果你渲染一个金属花瓶，它的光线反射效果会不同于毛绒玩具熊。

_纹理：_

- PBR使用纹理为物体表面添加细节，例如凹凸、划痕或小凹槽。这使得物体看起来更真实，因为在现实中，很少有东西是完全光滑的。

_真实性：_

- PBR的目标是使物体看起来尽可能接近现实。它通过考虑光线在现实中的表现方式、不同材质与光线的互动方式以及表面的微小瑕疵来实现这一点。

_光线的层次：_

- 想象一下你在看一杯水。PBR会尝试模拟光线穿过水的方式以及它如何可能扭曲你的视线。
- 它考虑了光线与物体不同部分的多层次互动，使渲染图像更加真实。

PBR还简化了工作流程。你无需手动调整大量参数来获得合适的外观，只需使用一组标准化的材质和照明模型。这样，过程变得更直观且更易操作。

现在，想象一下训练计算机视觉的AI模型。如果你在教计算机识别图像中的物体，拥有一组逼真且多样化的图像将会非常有帮助。PBR有助于生成看起来非常真实的合成数据，从而有效地训练计算机视觉模型。

有几种3D渲染引擎可以用于PBR，包括[Blender Cycles](https://www.blender.org)和[Unity](https://unity.com)。我们将重点介绍Blender，因为它是开源的，并且有大量关于Blender的资源。

## Blender

Blender是一款功能强大的开源3D计算机图形软件，用于创建动画电影、视觉效果、艺术作品、3D游戏等。它包含了广泛的功能，使其成为艺术家、动画师和开发者的多功能工具。让我们从渲染一张大象的合成图像的基本示例开始。

以下是基本步骤：

- 创建大象模型。下面展示的模型是使用[Metascan](https://metascan.ai)应用通过摄影测量创建的。摄影测量是一种将普通照片转化为3D模型的方法，就像从不同角度拍摄一堆玩具的照片，然后使用这些照片生成计算机版本。
- 创建背景——这是一个多步骤过程。详细说明见[此处](https://github.com/kfahn22/Synthetic-Data-Creation-in-Blender/tree/main/BACKGROUND)。
- 调整光照和相机位置。
- 固定大象的位置和旋转，使其适合框架（或相机视野）。

以下是使用Blender生成的大象图像：

![elephant image](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/rendered_elephant.png)

虽然还不完全是真实感图像，但已经足够接近，可以用于训练监测大象数量的模型。当然，要做到这一点，我们需要创建一个庞大的合成大象图像数据集！你可以使用Blender的Python环境[bpy](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html)渲染大量图像，随机化大象的位置和旋转。你还可以使用脚本进行分割、深度、法线和姿势估计的辅助工作。

太好了！我们该如何开始？

不幸的是，Blender的学习曲线相当陡峭。虽然每一步都不太复杂，但如果我们能在无需摸索的情况下渲染数据集，那该多好？幸运的是，有一个名为BlenderProc的库，包含了我们所需的所有脚本，用于渲染逼真的合成数据和注释，并且它是基于Blender构建的。

## BlenderProc

BlenderProc管道由Denninger等人在[BlenderProc](https://arxiv.org/abs/1911.01911)中引入，它是一个基于[Blender](https://www.blender.org)的模块化管道。它可用于生成多种用例中的图像，包括分割、深度、法线和姿势估计。

它专门为帮助生成真实感图像以训练卷积神经网络而创建，具有以下特点，使其成为合成数据生成的理想选择：

- 程序生成：通过程序技术实现复杂3D场景的自动创建和变化。
- 仿真：支持集成仿真，包括物理仿真，以增强真实感。
- 大规模生成：专为高效处理大规模场景生成设计，适用于多种应用。
- 自动化与可扩展性：
  - 脚本化：允许用户使用Python脚本自动生成过程，按照需求调整BlenderProc并配置参数。
  - 并行处理：支持并行处理以实现可扩展性，使生成大量场景变得高效。

你可以通过pip安装BlenderProc：

```bash
pip install blenderProc
```

或者，可以使用Git从GitHub克隆官方的[BlenderProc库](https://github.com/DLR-RM/BlenderProc)：

```bash
git clone https://github.com/DLR-RM/BlenderProc
```

BlenderProc必须在Blender的Python环境（bpy）中运行，因为这是访问Blender API的唯一方式。

```bash
blenderproc run <your_python_script>
```

你可以在Google Colab上查看这个笔记本，尝试BlenderProc的基本示例，参见[此处](https://github.com/DLR-RM/BlenderProc/tree/main/examples/basics)。以下是使用基础示例渲染的图像：

![colors](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/colors.png)
![normals](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/normals.png)
![depth](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/synthetic-data-creation-PBR/depth.png)

## Blender资源

- [用户手册](https://docs.blender.org/manual/en/latest/0)
- [Awesome-blender -- 丰富的资源列表](https://awesome-blender.netlify.app)
- [Blender YouTube频道](https://www.youtube.com/@BlenderOfficial)

### 以下视频解释了如何在Blender中渲染3D合成数据集：

<Youtube id="E1Pqpfg5kSo" />

### 以下视频解释了如何使用摄影测量创建3D对象：

<Youtube id="Pcqokf3PG_4" />

## 论文 / 博客

- [在Blender中开发多摄像头测量系统的数字孪生](https://iopscience.iop.org/article/10.1088/1361-6501/acc59e/pdf_)
- [使用Blender生成深度和法线图](https://www.saifkhichi.com/blog/blender-depth-map-surface-normals)
- [使用合成训练数据进行对象检测](https://medium.com/rowden/object-detection-with-synthetic-training-data-f6735a5a34bc)

## BlenderProc资源

- [BlenderProc Github仓库](https://github.com/DLR-RM/BlenderProc)
- [BlenderProc：通过真实感渲染缩小现实差距](https://elib.dlr.de/139317/1/denninger.pdf)
- [文档](https://dlr-rm.github.io/BlenderProc/)

### 以下视频提供了BlenderProc管道的概述：

<Youtube id="1AvY_iS6xQA" />

## 

论文

- [3D Menagerie：动物的3D形状和姿势建模]()
- [以假乱真：仅使用合成数据进行自然环境中的面部分析]()
- [在高度混乱的拾取箱中进行对象检测和基于自动编码器的6D姿势估计](https://arxiv.org/pdf/2106.08045.pdf)
- [从合成动物中学习](https://arxiv.org/abs/1912.08265)