# 欢迎来到社区计算机视觉课程

亲爱的学习者，

欢迎参加**社区驱动的计算机视觉课程**。计算机视觉正以多种方式改变我们的世界，从面部识别解锁手机到分析医疗图像进行疾病检测、监测野生动物、生成新图像等。让我们一起深入探索这个充满魅力的计算机视觉世界！

在整个课程中，我们将涵盖计算机视觉从基础到最新进展的各个方面。课程结构包括各种基础主题，适合所有人，易于理解和学习。我们非常高兴能有您加入这一激动人心的旅程！

在本页中，您可以找到如何加入学习者社区、提交作业获取证书的方式以及更多有关课程的详细信息！

## 作业 📄

要获得完成课程的认证，您需要完成以下作业：

1. 训练/微调模型
2. 构建一个应用并在 Hugging Face Spaces 上托管

### 训练/微调模型

在 Notebooks/Vision Transformers 部分有一些笔记本。目前，我们提供了对象检测、图像分割和图像分类的笔记本。您可以在 🤗 Hub 上现有的数据集上训练模型，或者将数据集上传到数据集库并在该数据集上训练模型。

模型库需要包含以下内容：

1. 完整的模型卡，更多信息可以查看[此处](https://huggingface.co/docs/hub/en/model-cards)。
2. 如果您使用 Transformers 训练了一个模型并推送到 Hub，模型卡将会自动生成。在这种情况下，请编辑模型卡并补充详细信息。
3. 在模型卡中添加数据集的 ID，以便将模型库链接到数据集库。

### 创建空间

在此作业部分，您将为您的计算机视觉模型构建一个基于 Gradio 的应用，并在 🤗 Spaces 上分享。可以通过以下资源了解更多：

- [Gradio 入门](https://huggingface.co/learn/nlp-course/chapter9/1?fw=pt#introduction-to-gradio)
- [如何在 🤗 Spaces 上分享您的应用](https://huggingface.co/learn/nlp-course/chapter9/4?fw=pt)

## 认证 🥇

完成作业后——训练/微调模型并创建空间——请填写[表格](https://forms.gle/isiVSw59oiiHP6pN9)，提供您的姓名、电子邮件地址以及您的模型和空间库链接，即可获得您的证书。

## 加入社区！

我们邀请您加入[我们活跃且支持的 Discord 社区](http://hf.co/join/discord)，这里每天都有有趣的讨论和共同兴趣的分享，同时也是本课程的起点。在这里，您将找到可以与您交换想法和资源的同伴。这是一个协作、获得反馈和提出问题的来源！

加入社区也是激励自己跟随课程的一个好方法。我们的社区随着 AI 的不断进步，讨论的质量和观点的多样性也在不断提升。成为会员后，您将有机会与其他课程参与者联系，交流想法，与他人合作。此外，课程的贡献者也活跃在 Discord 上，随时为您提供帮助。立即加入我们吧！

## 计算机视觉频道

我们的 Discord 服务器上有许多专注于各种主题的频道。在这里，您将找到关于论文讨论、活动组织、项目和想法分享、头脑风暴等的讨论。

作为计算机视觉课程的学习者，您可能会对以下频道特别感兴趣：

- `#computer-vision`: 与计算机视觉相关的所有内容的综合频道
- `#cv-study-group`: 用于交换想法、提出关于特定帖子的疑问并开始讨论的地方
- `#3d`: 专门讨论 3D 计算机视觉方面的频道

如果您对生成式 AI 感兴趣，我们还邀请您加入 Diffusion Models 相关的所有频道：#core-announcements, #discussions, #dev-discussions 和 #diff-i-made-this。

## 您将学到的内容

课程由理论部分、动手教程和有趣的挑战组成。

- **理论部分**：详细讲解计算机视觉的理论原理，并配有实际示例。
- **动手教程**：您将学习如何使用 Google Colab 笔记本训练和应用关键的计算机视觉模型。

在整个课程中，我们将涵盖计算机视觉从基础到最新进展的各个方面。课程结构包含了多种基础主题，让您全面了解当今计算机视觉的影响力。

## 先决条件

在开始本课程之前，确保您具备一定的 Python 编程经验，并了解 Transformer、机器学习和神经网络的基础知识。如果这些对您来说是新的，可以考虑先复习 [Hugging Face NLP 课程的第一单元](https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt)。虽然掌握预处理技术和诸如卷积等数学操作的知识会有所帮助，但它们并非必需。

## 课程结构

课程分为多个单元，涵盖了基础知识，并深入探索了最先进的模型。

- **单元 1 - 计算机视觉基础**：介绍计算机视觉的基本概念、该领域的基础知识和应用。探索图像基础、成像原理和预处理，以及特征提取的关键方面。
- **单元 2 - 卷积神经网络 (CNNs)**：深入理解 CNNs 的结构、关键概念和常见的预训练模型。学习如何使用迁移学习和微调技术将 CNNs 应用于各种任务。
- **单元 3 - 视觉 Transformer**：探索 Transformer 架构在计算机视觉中的应用，了解其与 CNN 的比较。熟悉常见的视觉 Transformer，如 Swin、DETR 和 CVT，并掌握迁移学习和微调的技术。
- **单元 4 - 多模态模型**：通过探索图像到文本和文本到图像等多模态任务，理解文本与视觉的融合。学习如 CLIP 及其衍生模型（如 GroupViT、BLIPM、Owl-VIT）等多模态模型，并掌握多模态任务的迁移学习技巧。
- **单元 5 - 生成模型**：探讨生成模型，包括 GANs、VAEs 和扩散模型。了解它们的差异以及在文本到图像、图像到图像和修复等任务中的应用。
- **单元 6 - 基本计算机视觉任务**：涵盖图像分类、对象检测和分割等基本任务及其使用的模型（如 YOLO、SAM）。了解这些任务的指标和实际应用。
- **单元 7 - 视频和视频处理**：研究视频的特性、视频处理的作用及其相对于图像处理的挑战。探索时间连续性、运动估计以及视频处理的实际应用。
- **单元 8 - 3D 视觉、场景渲染和重建**：深入了解三维视觉的复杂性，探讨场景渲染和重建中的 Nerf 和 GQN 等概念。理解 3D 视觉在计算机视觉中的挑战和应用，以及它如何提供更全面的空间信息视角。
- **单元 9 - 模型优化**：探讨模型优化的关键方面。涵盖模型压缩、部署考量以及工具和框架的使用。涉及蒸馏、剪枝和 TinyML 等主题以实现高效的模型部署。
- **单元 10 - 合成数据生成**：通过深度生成模型了解合成数据生成的重要性。探索点云和扩散模型等方法，调查主要合成数据集及其在计算机视觉中的应用。
- **单元 11 - 零样本计算机视觉**：深入零样本学习的领域，涵盖泛化、迁移学习及其在零样本识别和图像分割等任务中的应用。探讨零样本学习与各个计算机视觉领域中的迁移学习之间的关系。
- **单元 12 - 计算机视觉的伦理与偏见**：理解计算机视觉中特有的伦理考量。探讨伦理重要性、偏见如何渗入 AI 模型以及这些领域中常见的偏见类型。学习如何进行偏见评估和减轻策略，强调 AI 技术的负责任开发与部署。
- **单元 13 - 展望和新兴趋势**：探索当前趋势和新兴架构。深入了解创新方法，如 Retentive Network、Hiera、Hyena、I-JEPA 和 Retention Vision Models。

## 认识我们的团队

这门课程是由 Hugging Face 社区倾情打造的💜！通过在 [GitHub 上贡献](https://github.com/johko/computer-vision-course)加入我们吧。我们的目标是创建一个适合初学者的计算机视觉课程，并能为他人提供资源。来自世界各地的 60 多人齐心协力，共同完成了这个项目。在这里我们向他们致以谢意：

**Unit 1 - Fundamentals of Computer Vision**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [Ameed Taylor](https://github.com/atayloraerospace), [Sergio Paniego](https://github.com/sergiopaniego)
- Writers: [Seshu Pavan Mutyala](https://github.com/seshupavan), [Isabella Bicalho-Frazeto](https://github.com/bellabf), [Aman Kapoor](https://github.com/aman06012003), [Tiago Comassetto Fróes](https://github.com/froestiago), [Aditya Mishra](https://github.com/adityaiiitr), [Kerem Delikoyun](https://github.com/krmdel), [Ker Lee Yap](https://github.com/klyap), [Kathy Fahnline](https://github.com/kfahn22), [Ameed Taylor](https://github.com/atayloraerospace)

**Unit 2 - Convolutional Neural Networks (CNNs)**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [Mohammed Hamdy](https://github.com/mmhamdy), [Sezan](https://github.com/sezan92), [Joshua Adrian Cahyono](https://github.com/JvThunder), [Murtaza Nazir](https://github.com/themurtazanazir), [Albert Kao](https://github.com/albertkao227), [Sitam Meur](https://github.com/sitamgithub-MSIT), [Antonis Stellas](https://github.com/AntonisCSt), [Sergio Paniego](https://github.com/sergiopaniego)
- Writers: [Emre Albayrak](https://github.com/emre570), [Caroline Shamiso Chitongo](https://github.com/ShamieCC), [Sezan](https://github.com/sezan92), [Joshua Adrian Cahyono](https://github.com/JvThunder), [Murtaza Nazir](https://github.com/themurtazanazir), [Albert Kao](https://github.com/albertkao227), [Isabella Bicalho-Frazeto](https://github.com/bellabf), [Aman Kapoor](https://github.com/aman06012003), [Sitam Meur](https://github.com/sitamgithub-MSIT)

**Unit 3 - Vision Transformers**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [Mohammed Hamdy](https://github.com/mmhamdy), [Ameed Taylor](https://github.com/atayloraerospace), [Sezan](https://github.com/sezan92)
- Writers: [Surya Guthikonda](https://github.com/SuryaKrishna02), [Ker Lee Yap](https://github.com/klyap), [Anindyadeep Sannigrahi](https://bento.me/anindyadeep), [Celina Hanouti](https://github.com/hanouticelina), [Malcolm Krolick](https://github.com/Mkrolick), [Alvin Li](https://github.com/alvanli), [Shreyas Daniel Gaddam](https://shreydan.github.io), [Anthony Susevski](https://github.com/asusevski), [Alan Ahmet](https://github.com/alanahmet), [Ghassen Fatnassi](https://github.com/ghassen-fatnassi)

**Unit 4 - Multimodal Models**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [Snehil Sanyal](https://github.com/snehilsanyal), [Mohammed Hamdy](https://github.com/mmhamdy), [Charchit Sharma](https://github.com/charchit7), [Ameed Taylor](https://github.com/atayloraerospace), [Isabella Bicalho-Frazeto](https://github.com/bellabf)
- Writers: [Snehil Sanyal](https://github.com/snehilsanyal), [Surya Guthikonda](https://github.com/SuryaKrishna02), [Mateusz Dziemian](https://github.com/mattmdjaga), [Charchit Sharma](https://github.com/charchit7), [Evstifeev Stepan](https://github.com/minemile), [Jeremy Kespite](https://github.com/jeremy-k3/), [Isabella Bicalho-Frazeto](https://github.com/bellabf), [Pedro Gabriel Gengo Lourenco](https://github.com/pedrogengo)

**Unit 5 - Generative Models**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [William Bonvini](https://github.com/WilliamBonvini), [Mohammed Hamdy](https://github.com/mmhamdy), [Ameed Taylor](https://github.com/atayloraerospace)-
- Writers: [Jeronim Matijević](github.com/jere357), [Mateusz Dziemian](https://github.com/mattmdjaga), [Charchit Sharma](https://github.com/charchit7), [Muhammad Waseem](https://github.com/hwaseem04)

**Unit 6 - Basic Computer Vision Tasks**

- Reviewers: [Adhi Setiawan](https://github.com/adhiiisetiawan)
- Writers: [Adhi Setiawan](https://github.com/adhiiisetiawan), [Bastien Pouëssel](https://github.com/Skower)

**Unit 7 - Video and Video Processing**

- Reviewers: [Ameed Taylor](https://github.com/atayloraerospace), [Isabella Bicalho-Frazeto](https://github.com/bellabf)
- Writers: [Diwakar Basnet](https://github.com/DiwakarBasnet), [Chulhwa Han](https://github.com/cjfghk5697)

**Unit 8 - 3D Vision, Scene Rendering, and Reconstruction**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [William Bonvini](https://github.com/WilliamBonvini), [Mohammed Hamdy](https://github.com/mmhamdy), [Adhi Setiawan](https://github.com/adhiiisetiawan), [Ameed Taylor](https://github.com/atayloraerospace0)
- Writers: [John Fozard](https://github.com/jfozard), [Vasu Gupta](https://github.com/vasugupta9), [Psetinek](https://github.com/psetinek)

**Unit 9 - Model Optimization**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [Mohammed Hamdy](https://github.com/mmhamdy), [Adhi Setiawan](https://github.com/adhiiisetiawan), [Ameed Taylor](https://github.com/atayloraerospace)
- Writer: [Adhi Setiawan](https://github.com/adhiiisetiawan)

**Unit 10 - Synthetic Data Creation**

- Reviewers: [Mohammed Hamdy](https://github.com/mmhamdy), [Ameed Taylor](https://github.com/atayloraerospace), [Bhavesh Misra](https://github.com/Zekrom-7780)
- Writers: [William Bonvini](https://github.com/WilliamBonvini), [Alper Balbay](https://github.com/alperiox), [Madhav Kumar](https://github.com/miniMaddy), [Bhavesh Misra](https://github.com/Zekrom-7780), [Kathy Fahnline](https://github.com/kfahn22)

**Unit 11 - Zero Shot Computer Vision**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [Mohammed Hamdy](https://github.com/mmhamdy), [Albert Kao](https://github.com/albertkao227), [Isabella Bicalho-Frazeto](https://github.com/bellabf)
- Writers: [Mohammed Hamdy](https://github.com/mmhamdy), [Albert Kao](https://github.com/albertkao227)

**Unit 12 - Ethics and Biases in Computer Vision**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [Mohammed Hamdy](https://github.com/mmhamdy), [Charchit Sharma](https://github.com/charchit7), [Adhi Setiawan](https://github.com/adhiiisetiawan), [Ameed Taylor](https://github.com/atayloraerospace), [Bhavesh Misra](https://github.com/Zekrom-7780)
- Writers: [Snehil Sanyal](https://github.com/snehilsanyal), [Bhavesh Misra](https://github.com/Zekrom-7780)

**Unit 13 - Outlook and Emerging Trends**

- Reviewers: [Ratan Prasad](https://github.com/ratan), [Ameed Taylor](https://github.com/atayloraerospace), [Mohammed Hamdy](https://github.com/mmhamdy)
- Writers: [Farros Alferro](https://github.com/farrosalferro), [Mohammed Hamdy](https://github.com/mmhamdy), [Louis Ulmer](https://github.com/lulmer), [Dario Wisznewer](https://github.com/dariowsz), [gonzachiar](https://github.com/gonzachiar)

**Organisation Team**
[Merve Noyan](https://github.com/merveenoyan), [Adam Molnar](https://github.com/lunarflu), [Johannes Kolbe](https://github.com/johko)



我们很高兴您加入，让我们开始吧！