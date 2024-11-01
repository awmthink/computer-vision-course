# 特征匹配

如何将一张图像中的检测特征与另一张图像中的特征匹配？特征匹配涉及比较不同图像中的关键属性，以寻找相似性。特征匹配在许多计算机视觉应用中都很有用，包括场景理解、图像拼接、对象跟踪和模式识别。

## 暴力搜索

想象你有一个巨大的拼图盒子，你正在寻找一个特定的拼图块来匹配你的拼图。这类似于在图像中搜索匹配的特征。没有任何特殊策略，你决定逐一检查每一块，直到找到合适的。这种直接的方法就是暴力搜索。暴力搜索的优势在于其简单性。你不需要任何特殊技巧——只需要耐心。然而，若拼图块很多，可能会非常耗时。在特征匹配的背景下，这种暴力方法类似于将一张图像中的每一个像素与另一张图像中的每一个像素进行比较，以查看它们是否匹配。这种方法是全面的，可能需要很多时间，尤其是对于大图像而言。

现在我们对如何找到暴力匹配有了直观的理解，让我们深入探讨算法。我们将使用上一章学到的描述符在两张图像中找到匹配特征。

首先，安装并加载库。

```bash
!pip install opencv-python
```

```python
import cv2
import numpy as np
```

**使用 SIFT 的暴力匹配**

首先初始化 SIFT 检测器。

```python
sift = cv2.SIFT_create()
```

使用 SIFT 查找关键点和描述符。

```python
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
```

使用 k 近邻算法找到匹配。

```python
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
```

应用比值测试筛选出最佳匹配。

```python
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
```

绘制匹配结果。

```python
img3 = cv2.drawMatchesKnn(
    img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/feature-extraction-feature-matching/SIFT.png" alt="SIFT">
</div>

**使用 ORB（二进制）描述符的暴力匹配**

初始化 ORB 描述符。

```python
orb = cv2.ORB_create()
```

查找关键点和描述符。

```python
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
```

由于 ORB 是一种二进制描述符，我们使用[汉明距离](https://www.geeksforgeeks.org/hamming-distance-two-strings/)来衡量两个等长字符串之间的差异。

```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
```

接下来找到匹配。

```python
matches = bf.match(des1, des2)
```

按距离顺序对它们进行排序。

```python
matches = sorted(matches, key=lambda x: x.distance)
```

绘制前 n 个匹配。

```python
img3 = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:n],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
```

**近似最近邻快速库 (FLANN)**

Muja 和 Lowe 提出的 [FLANN](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_visapp09.pdf) 是一种快速近似最近邻算法。为了说明 FLANN，我们继续使用拼图的例子。设想一个散落着数百块拼图的大拼图，你的目标是将这些拼图块按匹配度进行组织。FLANN 使用一些聪明的技巧来快速找出最可能匹配的拼图块，而不是随机尝试。FLANN 在底层使用一种称为 k-D 树的结构。可以将其视为以特殊方式组织拼图块。在 k-D 树的每个节点中，FLANN 将具有相似特征的拼图块放在一起。这样，当你寻找匹配时，可以快速检查最可能具有相似特征的拼图块。

首先，为 SIFT 或 SURF 创建指定算法的字典，如下所示。

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
```

对于 ORB，使用论文中的参数。

```python
FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2
)
```

我们还创建一个字典来指定要访问的最大叶节点数。

```python
search_params = dict(checks=50)
```

初始化 SIFT 检测器。

```python
sift = cv2.SIFT_create()
```

使用 SIFT 查找关键点和描述符。

```python
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
```

定义 FLANN 参数。树的数量是希望的 bin 数量。

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)
```

我们只绘制好的匹配，因此创建一个掩码。

```python
matchesMask = [[0, 0] for i in range(len(matches))]
```

可以执行比值测试来确定好的匹配。

```python
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
```

现在我们来可视化匹配。

```python
draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=cv2.DrawMatchesFlags_DEFAULT,
)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
```

![FLANN](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/feature-extraction-feature-matching/FLANN.png)

## 基于 Transformer 的局部特征匹配 (LoFTR)

LoFTR 由 Sun 等人提出于 [LoFTR: Detector-Free Local Feature Matching with Transformers](https://arxiv.org/pdf/2104.00680.pdf)。LoFTR 使用基于学习的方法进行特征匹配，而非传统特征检测器。

我们用拼图的例子来简化说明。LoFTR 会在每张图像中寻找特定的关键点或特征。它还会处理旋转或缩放的变化。如果一个特征被旋转或调整大小，LoFTR 仍然能识别出来。LoFTR 在匹配特征时，会赋予一个相似度分数以指示特征对齐的程度。更高的分数表示更好的匹配。此外，LoFTR 能够处理一定的变换不变性，这使得它在处理在不同条件下拍摄的图像时非常重要。LoFTR 的鲁棒性特征使其在图像拼接等任务中很有价值，因为拼接需要通过识别和连接公共特征来无缝地结合多张图像。

我们可以使用 [Kornia](https://github.com/kornia/kornia) 在两张图像中使用 LoFTR 查找匹配特征。

```bash
!pip install kornia  kornia-rs  kornia_moons opencv-python --upgrade
```

导入必要的库。

```python
import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches
```

加载并调整图像大小。

```python
from kornia.feature import LoFTR

img1 = K.io.load_image(image1.jpg, K.io.ImageLoadType.RGB32)[None, ...]
img2 = K.io.load_image(image2.jpg, K.io.ImageLoadType.RGB32)[None, ...]

img1 = K.geometry.resize(img1, (512, 512), antialias=True)
img2 = K.geometry.resize(img2, (512, 512), antialias=True)
```

指示图像是否为“室内”或“室

外”图像。

```python
matcher = LoFTR(pretrained="outdoor")
```

LoFTR 仅适用于灰度图像，因此将图像转换为灰度。

```python
input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),
    "image1": K.color.rgb_to_grayscale(img2),
}
```

执行推理。

```python
with torch.inference_mode():
    correspondences = matcher(input_dict)
```

使用随机采样一致性 ([RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)) 清理对应关系，以处理数据中的噪声或离群值。

```python
mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0
```

最后，我们可以可视化匹配结果。

```python
draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={
        "inlier_color": (0.1, 1, 0.1, 0.5),
        "tentative_color": None,
        "feature_color": (0.2, 0.2, 1, 0.5),
        "vertical": False,
    },
)
```

最佳匹配以绿色显示，而不确定的匹配显示为蓝色。

![LoFTR](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/feature-extraction-feature-matching/LoFTR.png)

## 资源和进一步阅读

- [FLANN Github](https://github.com/flann-lib/flann)
- [Image Matching Using SIFT, SURF, BRIEF and ORB: Performance Comparison for Distorted Images](https://arxiv.org/pdf/1710.02726.pdf)
- [ORB (Oriented FAST and Rotated BRIEF) tutorial](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)
- [Kornia tutorial on Image Matching](https://kornia.github.io/tutorials/nbs/image_matching.html)
- [LoFTR Github](https://github.com/zju3dv/LoFTR)
- [OpenCV Github](https://github.com/opencv/opencv-python)
- [OpenCV Feature Matching Tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [OpenGlue: Open Source Graph Neural Net Based Pipeline for Image Matching](https://arxiv.org/abs/2204.08870)