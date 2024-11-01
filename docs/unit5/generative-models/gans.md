# 生成对抗网络

## 简介
生成对抗网络（Generative Adversarial Networks，GANs）是一类深度学习模型，由[Ian Goodfellow](https://scholar.google.ca/citations?user=iYN86KEAAAAJ&hl=en)及其同事在2014年提出。GANs的核心思想是训练一个生成器网络来生成与真实数据无法区分的数据，同时训练一个判别器网络来区分真实数据和生成数据。
* **架构概述：** GANs由两个主要组成部分构成：`生成器`和`判别器`。
* **生成器：** 生成器以随机噪声$z$作为输入，生成合成数据样本。其目标是创建足够逼真的数据，以迷惑判别器。
* **判别器：** 判别器类似于一个侦探，评估给定样本是真实的（来自实际数据集）还是虚假的（由生成器生成）。其目标是不断提高区分真实样本和生成样本的准确性。

网上常见的一个类比是画家/伪造者（生成器）试图伪造画作，而艺术鉴定家/评论家（判别器）则试图识别伪造作品的缺陷。

![Lilian Weng GAN Figure](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/generative_models/GAN.png)

## GANs vs VAEs
GANs和VAEs都是机器学习中常见的生成模型，但它们各有优劣。哪个更“好”取决于具体任务和需求。以下是它们优缺点的对比：
* **图像生成：**
    - **GANs:**
        * **优点：** 能生成更高质量的图像，特别适用于具有清晰细节和逼真纹理的复杂数据。
        * **缺点：** 训练难度较大，容易出现不稳定性。
        * **示例：** 由GAN生成的卧室图像可能与真实图像难以区分，而由VAE生成的卧室可能显得模糊或光照不自然。
        ![Example of GAN-Generated bedrooms taken from Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, 2015](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/generative_models/bedroom.png)
    - **VAEs:**
        * **优点：** 训练更容易且更加稳定。
        * **缺点：** 生成的图像可能模糊，细节较少，甚至可能有不现实的特征。
* **其他任务：**
    - **GANs:**
        * **优点：** 可用于超分辨率和图像到图像的转换任务。
        * **缺点：** 在需要平滑过渡的数据点任务中可能并非最佳选择。
    - **VAEs:**
        * **优点：** 广泛应用于图像去噪和异常检测任务。
        * **缺点：** 在需要高质量图像生成的任务中，可能不如GANs有效。

以下是关键差异的总结表格：

| 特性          | GANs            | VAEs           |
|---------------|-----------------|----------------|
| 图像质量      | 高              | 低             |
| 训练难度      | 更困难          | 更容易         |
| 稳定性        | 不稳定          | 更稳定         |
| 应用          | 图像生成，超分辨率，图像到图像转换 | 图像去噪，异常检测，信号分析 |

最终的选择取决于具体需求和优先级。如果需要高质量图像用于生成逼真的人脸或风景等任务，GAN可能是更好的选择；但如果需要一个训练更容易且更稳定的模型，VAE可能是更好的选择。

## GAN的训练
训练GANs涉及一个独特的对抗过程，其中生成器和判别器进行“猫捉老鼠”游戏。

* **对抗训练过程：** 生成器和判别器同时训练。生成器旨在生成与真实数据无异的数据，而判别器则努力提高区分真实和虚假样本的能力。
* **目标函数：** 训练过程由一种最小-最大博弈类型的目标函数引导，用于优化生成器和判别器。生成器旨在最小化判别器正确分类生成样本为假样本的概率，而判别器则试图最大化这一概率。该目标函数表示如下：
$$\min_G \max_D L(D, G)=\mathbb{E}_{x \sim p_{r}(x)} [\log D(x)] + \mathbb{E}_{x \sim p_g(x)} [\log(1 - D(x))]$$
在这里，判别器试图最大化此损失函数，而生成器试图最小化它，因此具有对抗性。
* **迭代改进：** 随着训练进展，生成器逐渐擅长生成逼真的样本，而判别器变得更有辨别力。该对抗循环持续进行，直至生成器生成的数据几乎无法与真实数据区分。

## 参考文献：
1. [Lilian Weng的关于GAN的优秀博客](https://lilianweng.github.io/posts/2017-08-20-gan/)
2. [GAN — 什么是生成对抗网络](https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09)
3. [VAE和GAN在图像生成方面的根本区别是什么？](https://ai.stackexchange.com/questions/25601/what-are-the-fundamental-differences-between-vae-and-gan-for-image-generation)
4. [GAN和VAE模型的问题](https://stats.stackexchange.com/questions/541775/issues-with-gan-and-vae-models)
5. [VAE与GAN在图像生成中的对比](https://www.baeldung.com/cs/vae-vs-gan-image-generation)
6. [扩散模型 vs. GANs vs. VAEs: 深度生成模型的比较](https://towardsai.net/p/machine-learning/diffusion-models-vs-gans-vs-vaes-comparison-of-deep-generative-models)