# 人工智能的伦理与偏见 🧑‍🤝‍🧑

我们希望你对 ImageNet Roulette 案例研究感兴趣，并了解到 AI 模型在实际应用中可能出现的问题。在本章节中，我们将通过另一个强大技术的例子来讨论其有趣的应用及其在未加监管情况下可能引发的伦理问题。首先，让我们简要总结一下 ImageNet Roulette 案例研究及其后果。
- ImageNet Roulette 是一个典型的 AI 模型失控的例子，原因在于其固有的偏见以及在标记和数据预处理阶段的疏忽。
- 该实验是人为设计的，目的是展示如果不加以控制会出现的问题。
- 该项目促使 ImageNet 团队对数据集进行大量修正，并采取适当的措施来减轻问题，比如面部模糊处理、去除有害和触发性的同义词集、移除相关图像等。
- 最终，该项目引发了一个持续的讨论，并推动了有关减轻风险的研究工作。

在我们讨论另一个强大技术的例子之前，让我们先退后一步，思考一些问题。总体而言，技术是好是坏？电力是好是坏？互联网总体来说是安全的还是有害的？等等。在开始我们的探讨时，记住这些问题。

## 深度伪造 🎥

想象一下，你是一位刚毕业的学生，想要学习深度学习。你报名参加了麻省理工学院的 "Introduction to Deep Learning (MIT 6.S191)" 课程。为了增加趣味性，课程团队发布了一段非常酷的视频，展示了深度学习的应用。点击这里观看视频：

<Youtube id="l82PxsKHxYc"/>
麻省理工学院 6.S191 课程的介绍视频，其中使用深度伪造技术模拟了奥巴马的欢迎致辞。

是的，介绍部分被制作得仿佛学生们受到奥巴马本人的欢迎。这是一种非常酷的应用，通过课程介绍展示深度生成模型的用例。对于初学者来说，这会非常吸引人，让他们对这项技术产生兴趣，人人都会想尝试制作这样的内容。毕竟，你可以使用一块高性能 GPU 在几分钟内轻松生成这样的图像和视频，然后分享表情包、帖子等。

让我们看看这种技术的另一个例子，但带来不同的影响。试想一下，如果在选举或战争期间，能制作出政治领袖或演员的深度伪造视频。这样的虚假视频可以被用来传播仇恨和错误信息，从而边缘化特定人群。即使这些人没有散布错误信息，视频本身也会引发极大的愤怒。这非常可怕。然而，主要问题在于，一旦错误信息传播开来，即使后来被证实视频是被操控的，分裂已经产生。因此，只有在伪造视频未被公开的情况下，才能避免这种伤害。这使得这种技术变得危险，但技术本身是安全还是有害的？技术本身并没有好坏之分，而是取决于它的使用（由谁使用以及用于何种目的），可能产生好的或坏的效果。

深度伪造是利用深度生成计算机视觉 (CV) 模型创建的合成媒体。你可以通过不同人物的图像操控图像，也可以生成视频。音频深度伪造是另一种技术，可以通过模仿特定对象的声音来补充 CV 技术。这只是深度伪造可能引发破坏的一个例子，实际上，其影响可能更为严重，对受害者的生活产生深远的影响。

## 什么是 AI 的伦理与偏见？

从前面的例子中，可以注意到该技术的一些关键点：
- 在使用图像/视频进行操控和生成新媒体之前，需要获得对象的同意。
- 支持创建可操控合成媒体的算法。
- 用于检测此类合成媒体的算法。
- 了解这些算法及其后果的必要性。

<Tip>

💡请查看 [The Consentful Tech Project](https://www.consentfultech.io/)。该项目旨在提高人们的意识，制定策略，并分享帮助人们合意地构建和使用技术的技能。
</Tip>

现在让我们基于这些示例正式定义一些术语。那么什么是伦理与偏见？伦理可以简单地定义为帮助我们区分对错的一套道德原则。而 AI 伦理可以定义为在开发和使用 AI 过程中采用广泛接受的对错标准来指导道德行为的一套价值观、原则和技术。AI 伦理是一个跨学科领域，研究如何优化 AI 的积极影响，同时减少风险和不利结果。该领域涉及各种利益相关者：
- **研究人员、工程师、开发者和 AI 伦理学家：**负责模型、算法、数据集开发和管理的人。
- **政府机构、法律部门（如律师）：**负责监管 AI 伦理发展的机构和人员。
- **公司和组织：**在提供 AI 产品和服务方面处于前沿的利益相关者。
- **公民：**在日常生活中使用 AI 服务和产品并受到该技术影响的人。

AI 中的偏见指的是算法输出中的偏见，这可能是由于模型开发或训练数据中的假设所导致的。这些假设源自负责开发的人的固有偏见，导致 AI 模型和算法反映出这些偏见。这些偏见会破坏伦理发展或原则，因此需要关注并采取措施加以减轻。我们将在本单元的后续章节中详细讨论偏见、偏见的类型、评估以及减轻方法（重点在 CV 模型）。为了更好地理解 AI 伦理，我们将深入探讨 AI 伦理原则。

## AI 伦理原则 🤗 🌎

### 阿西莫夫的机器人三定律 🤖 

有许多历史性作品涉及到技术伦理的发展。最早的作品可以追溯到著名科幻作家艾萨克·阿西莫夫。他考虑到自主 AI 代理的潜在风险，提出了机器人三定律。这三条定律为：
- 机器人不得伤害人类，或因不作为而使人类受到伤害。
- 机器人必须服从人类的命令，但当该命令与第一条定律相抵触时例外。
- 机器人必须保护自己的存在，只要这种保护不与第一或第二定律相抵触。

### 阿西洛马 AI 原则 🧑🏻‍⚖️🧑🏻‍🎓🧑🏻‍💻

阿西莫夫的机器人定律是技术伦理的最早作品之一。2017 年，在加利福尼亚的阿西洛马会议中心举行了一次会议，讨论了 AI 对社会的影响。会议的成果是制定了负责任的 AI 开发指南。这些准则包含了 23 条原则，签署者约有 5,000 人，其中包括 844 位 AI 和机器人研究人员。

![阿西洛马 AI 原则](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/asilomar-ai.png)
阿西洛马 AI 原则：负责任的 AI 开发的 23 条原则

<Tip>

💡你可以在[这里](https://futureoflife.org/open-letter/ai-principles/)查看完整的 23 条阿西洛马 AI 原则及签署者。
</Tip>

这些原则是 AI 模型负责任开发和实施的指导。接下来，我们来看一下联合国教科文组织 (UNESCO) 最近关于 AI 伦理的工作。

### 联合国教科文组织的报告：关于人工智能伦理的建议 🧑🏼‍🤝‍🧑🏼🌐

联合国教科文组织在 2021 年 11 月提出了一个关于 AI 伦理的全球标准，名为 **“关于人工智能伦理的建议”**，193 个成员国采纳了该报告。之前的 AI 伦理准则在可操作性政策方面存在不足，但联合国教科文组织的最新报告允许政策制定者将核心原则转化为数据治理、环境、性别、健康等不同领域的实际行动。该建议的四大核心价值观是：
- **人权与人类尊严**：尊重、保护和促进人权、基本自由和人类尊严。
- **和平共处**：在和平、正义和互联的社会中生活。
- **确保多样性和包容性。**
- **环境和生态系统的繁荣。**

![AI 政策领域](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/ai_policy.png)
负责任 AI 开发的 11 个关键政策领域。

联合国教科文组织提出的 10 条以人权为核心的 AI 伦理原则如下：
- **比例性与不造成伤害：** AI 系统的使用不得超出实现合法目标的必要范围，且应进行风险评估以防止可能导致的伤害。
- **安全性与安全保障：** 应避免和解决

不必要的危害（安全风险）以及攻击漏洞（安全风险）。
- **隐私权与数据保护：** 隐私权应在 AI 生命周期内得到保护和促进，且应建立足够的数据保护框架。
- **多方利益相关者与适应性治理及合作：** 数据使用应尊重国际法和国家主权，且不同利益相关者的参与对 AI 治理的包容性方法至关重要。
- **责任与问责：** AI 系统应具备可审计性和可追溯性，且应有监督、影响评估、审计和尽职调查机制，以避免与人权规范和环境威胁相冲突。
- **透明性与可解释性：** AI 系统的伦理部署依赖于其透明性和可解释性（T&E），T&E 的水平应符合具体情况，因为 T&E 与隐私、安全等原则之间可能存在冲突。
- **人类监督与决定权：** 各成员国应确保 AI 系统不会取代人类的最终责任和问责。
- **可持续性：** 应根据 AI 技术对“可持续性”的影响进行评估，这是一组不断演变的目标，包括联合国的可持续发展目标。
- **意识与素养：** 应通过开放和可访问的教育、公民参与、数字技能和 AI 伦理培训、媒体和信息素养来提高公众对 AI 和数据的理解。
- **公平与非歧视：** AI 行为者应促进社会正义、公平和非歧视，并采取包容性的方法确保 AI 的益处惠及所有人。

<Tip>

💡如需阅读联合国教科文组织的完整报告“关于人工智能伦理的建议”，请访问[这里](https://unesdoc.unesco.org/ark:/48223/pf0000381137)。
</Tip>

在单元结尾，我们还将讨论 Hugging Face 如何确保 AI 的伦理实践。在下一章中，我们将进一步学习偏见的类型以及它们如何渗透到不同的 AI 模型中。