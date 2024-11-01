# 多模态物体检测 (OWL-ViT)
### 介绍

物体检测是计算机视觉中的一项关键任务，随着YOLO等模型的发展，取得了显著的进展（[原始论文](https://arxiv.org/abs/1506.02640)，[最新代码版本](https://github.com/ultralytics/ultralytics)）。然而，像YOLO这样的传统模型在检测训练数据集之外的物体时具有局限性。为了解决这一问题，AI社区开始研发能够识别更广泛物体的模型，从而诞生了类似CLIP的物体检测模型。

### OWL-ViT: 增强功能与能力
OWL-ViT在开放词汇物体检测领域实现了跨越式发展。它的训练阶段类似于CLIP，专注于使用对比损失的视觉和语言编码器。这个基础阶段使模型能够学习视觉和文本数据的共享表示空间。

#### 物体检测的微调
OWL-ViT的创新之处在于其用于物体检测的微调阶段。在这里，OWL-VIT没有使用CLIP中的token池化和最终投影层，而是通过线性投影每个输出token，以获得每个物体的图像嵌入。然后使用这些嵌入进行分类，而边框坐标则通过一个小型MLP从token表示中得出。这种方法使OWL-ViT能够检测物体及其在图像中的空间位置，相较于传统物体检测模型是一项显著的进步。

以下是OWL-ViT的预训练和微调阶段的图示：

![OWL-ViT 预训练和微调](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/owlvit_architecture.jpg)

#### 开放词汇检测
经过微调后，OWL-ViT在开放词汇物体检测方面表现出色。由于视觉和文本编码器的共享嵌入空间，它能够识别训练数据集中没有明确出现的物体。这一能力使得OWL-ViT可以使用图像和文本查询进行物体检测，增强了其多样性。

#### 示例应用
在实际应用中，通常使用文本作为查询，并将图像作为上下文。以下Python示例展示了如何使用transformers库运行OWL-ViT的推理。

```python
import requests
from PIL import Image, ImageDraw
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog", "remote control", "cat tail"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.Tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.1
)
i = 0  # 为第一个图像获取相应的文本查询的预测结果
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# 创建绘图对象
draw = ImageDraw.Draw(image)

# 绘制每个边框
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"检测到 {text[label]}，置信度 {round(score.item(), 3)}，位置 {box}"
    )
    # 在图像上绘制边框
    draw.rectangle(box, outline="red")

# 显示图像
image
```
图像效果应如下所示：
![OWL-ViT 示例](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/OWL_ViT_example.jpg)
该示例展示了OWL-ViT的一个简单应用，我们指定了可能存在于数据集中的对象（如猫）和更抽象的对象（如猫尾巴），这些通常不会出现在任何物体检测数据集中。这展示了OWL-ViT在开放词汇物体检测中的强大能力。

可以使用此代码尝试更复杂的示例，或者试试这个Gradio应用：

<iframe
	src="https://johko-OWL-ViT.hf.space"
	frameborder="0"
	width="850"
	height="450">
</iframe>

### 结论

OWL-ViT在物体检测中的方法代表了AI模型理解和交互视觉世界的显著转变。通过将语言理解与视觉感知相结合，它推动了物体检测的边界，使得模型更为准确和多功能，能够识别更广泛的物体。这种模型能力的进化对于需要对视觉场景进行细致理解的应用至关重要，尤其是在动态的现实环境中。

欲了解更深入的信息和技术细节，请参考 [OWL-VIT 论文](https://arxiv.org/abs/2205.06230)。有关 [OWL-ViT 2 模型](https://huggingface.co/docs/transformers/model_doc/owlv2) 的信息也可以在Hugging Face文档中找到。