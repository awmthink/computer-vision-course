# Vision Transformers 对象检测

本章节将介绍如何使用 Vision Transformers 来实现对象检测任务。我们将学习如何针对我们的用例微调现有的预训练对象检测模型。开始之前，建议访问这个 HuggingFace Space，您可以尝试与最终结果互动。

<iframe
	src="https://hf-vision-finetuning-demo-for-object-detection.hf.space/"
	frameborder="0"
	width="850"
	height="450">
</iframe>

## 介绍

![对象检测示例](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/object_detection_wiki.png)

对象检测是计算机视觉中的一项任务，涉及识别和定位图像或视频中的对象。它包含两个主要步骤：

- 首先，识别存在的对象类型（例如汽车、人物或动物）。
- 其次，通过绘制边界框确定其精确位置。

这些模型通常接收图像（静态图像或视频帧）作为输入，每幅图像中可能包含多个对象。例如，考虑包含汽车、人物、自行车等多个对象的图像。处理输入后，这些模型会生成一组数值，传达以下信息：

- 对象的位置（边界框的 XY 坐标）。
- 对象的类别。

对象检测有许多应用，其中一个显著的例子是自动驾驶领域。对象检测用于检测汽车周围的不同对象（如行人、路标、交通信号灯等），并成为决策的输入之一。

如需深入了解对象检测的详细内容，请查看我们专门的[章节](https://huggingface.co/learn/computer-vision-course/unit6/basic-cv-tasks/object_detection)🤗。

### 微调对象检测模型的必要性 🤔

这是一个很棒的问题。从头开始训练对象检测模型意味着：

- 不断重复已经完成的研究。
- 编写重复的模型代码、训练模型，并为不同的用例维护不同的代码库。
- 需要大量的实验并浪费资源。

与其进行这些重复操作，不如采用一个表现出色的预训练模型（一个在识别一般特征方面表现优异的模型），并调整或重新微调其权重（或部分权重），以适应特定的用例。我们认为，预训练模型已经学会了足够多的特征，可以在图像中定位和分类对象。因此，如果引入新对象，利用已经学习的特征和新的特征，可以通过少量时间和计算来训练相同的模型，从而开始检测这些新对象。

通过本教程的学习，您将能够创建完整的对象检测流水线（包括加载数据集、微调模型和进行推断）。

## 安装必要的库

首先进行安装。执行以下命令来安装必要的软件包。在本教程中，我们将使用 Hugging Face Transformers 和 PyTorch。

```bash
!pip install -U -q datasets transformers[torch] evaluate timm albumentations accelerate
```

## 场景

为了让本教程更具吸引力，让我们考虑一个现实的示例：建筑工人需要在施工区域保持最高的安全性。基本的安全协议要求随时佩戴头盔。由于有很多建筑工人，很难时刻监控每个人。

但如果我们能够拥有一个可以实时检测是否佩戴头盔的摄像系统，那就非常理想了，对吗？

因此，我们将微调一个轻量级的对象检测模型，以实现这一目标。让我们开始吧。


### 数据集

在上述场景中，我们将使用[Northeaster University China](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/7CBGOS)提供的[hardhat](https://huggingface.co/datasets/hf-vision/hardhat)数据集。我们可以使用 🤗 `datasets` 下载并加载此数据集。

```python
from datasets import load_dataset

dataset = load_dataset("anindya64/hardhat")
dataset
```

这将提供以下数据结构：

```
DatasetDict({
    train: Dataset({
        features: ['image', 'image_id', 'width', 'height', 'objects'],
        num_rows: 5297
    })
    test: Dataset({
        features: ['image', 'image_id', 'width', 'height', 'objects'],
        num_rows: 1766
    })
})
```

以上是一个[DatasetDict](https://huggingface.co/docs/datasets/v2.17.1/en/package_reference/main_classes#datasets.DatasetDict)，这是一个包含整个数据集（按训练集和测试集划分）的高效字典结构。如您所见，在每个划分（训练集和测试集）下，我们有`features`和`num_rows`。在`features`中，有`image`（一个[Pillow对象](https://realpython.com/image-processing-with-the-python-pillow-library/)）、图像的ID、高度和宽度以及`objects`。  
现在让我们看看每个数据点（在训练/测试集中）是什么样的。要做到这一点，运行以下代码：

```python
dataset["train"][0]
```

这将生成以下结构：

```
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375>,
 'image_id': 1,
 'width': 500,
 'height': 375,
 'objects': {'id': [1, 1],
  'area': [3068.0, 690.0],
  'bbox': [[178.0, 84.0, 52.0, 59.0], [111.0, 144.0, 23.0, 30.0]],
  'category': ['helmet', 'helmet']}}
```

如您所见，`objects`是另一个字典，包含对象ID（即类别ID）、对象的面积以及边界框坐标(`bbox`)和类别（标签）。以下是每个键和数据元素值的详细解释。

- `image`: 这是一个Pillow图像对象，可直接在加载路径之前查看图像。
- `image_id`: 表示图像在训练文件中的编号。
- `width`: 图像的宽度。
- `height`: 图像的高度。
- `objects`: 另一个包含注释信息的字典。包含以下内容：
    - `id`: 一个列表，列表的长度表示对象的数量，每个值表示类索引。
    - `area`: 对象的面积。
    - `bbox`: 表示对象的边界框坐标。
    - `category`: 对象的类（字符串）。

现在我们来正确提取训练和测试样本。在本教程中，我们有大约5000个训练样本和1700个测试样本。

```python
# 首先，提取训练集和测试集

train_dataset = dataset["train"]
test_dataset = dataset["test"]
```

现在我们知道样本数据点的内容，让我们开始绘制该样本。我们将首先绘制图像，然后绘制相应的边界框。

以下是我们要做的步骤：

1. 获取图像及其相应的高度和宽度。
2. 创建一个可在图像上绘制文本和线条的绘制对象。
3. 从样本中获取注释字典。
4. 遍历注释字典。
5. 对于每个对象，获取边界框坐标，即x（水平起始位置）、y（垂直起始位置）、w（边界框的宽度）、h（边界框的高度）。
6. 如果边界框的度量是归一化的，则进行缩放，否则保持不变。
7. 最后绘制矩形和类别文本。

```python 
import numpy as np
from PIL import Image, ImageDraw


def draw_image_from_idx(dataset, idx):
    sample = dataset[idx]
    image = sample["image"]
    annotations = sample["objects"]
    draw = ImageDraw.Draw(image)
    width, height = sample["width"], sample["height"]

    for i in range(len(annotations["id"])):
        box = annotations["bbox"][i]
        class_idx = annotations["id"][i]
        x, y, w, h = tuple(box)
        if max(box) > 1.0:
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
        else:
            x1 = int(x * width)
            y1 = int(y * height)
            x2 = int((x + w) * width)
            y2 = int((y + h) * height)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=1)
        draw.text((x1, y1), annotations["category"][i], fill="white")
    return image


draw_image_from_idx(dataset=train_dataset, idx=10)
```

我们有一个函数来绘制单个图像，接下来编写一个简单的函数来使用上述代码绘制多个图像。这样可以帮助我们进行一些分析。

```python
import matplotlib.pyplot as plt


def plot_images(dataset, indices):
    """
    绘制图像及其注释。
    """
    num_rows = len(indices) // 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    for i, idx in enumerate(indices):
        row = i // num_cols
        col = i % num_cols

        # 绘制图像
        image = draw_image_from_idx(dataset, idx)

        # 在对应的子图上显示图像
        axes[row, col].imshow(image)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


# 现在使用该函数绘制图像

plot_images(train_dataset, range(9))
```

运行此函数将生成一个漂亮的拼图，如下所示。

![input-image-plot](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/object_detection_train_image_with_annotation_plots.png)

## AutoImageProcessor

在对模型进行微调之前，我们必须对数据进行预处理，以确保与预训练时的方法完全匹配。HuggingFace 的 [AutoImageProcessor](https://huggingface.co/docs/transformers/v4.36.0/en/model_doc/auto#transformers.AutoImageProcessor) 负责处理图像数据，以生成 `pixel_values`、`pixel_mask` 和 `labels`，供 DETR 模型进行训练。

现在，让我们从我们希望用来微调模型的相同检查点实例化图像处理器。

```python
from transformers import AutoImageProcessor

checkpoint = "facebook/detr-resnet-50-dc5"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

## 数据集预处理

在将图像传递给 `image_processor` 之前，让我们还对图像及其对应的边界框应用不同类型的增强。

简单来说，增强是一组随机转换，如旋转、调整大小等。这些转换被应用于图像，以获得更多样本，并使视觉模型在不同图像条件下更加健壮。我们将使用 [albumentations](https://github.com/albumentations-team/albumentations) 库来实现这一点。它允许您创建图像的随机变换，从而增加训练样本数量。

```python
import albumentations
import numpy as np
import torch

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)
```

一旦我们初始化了所有的转换，我们需要创建一个函数来格式化注释，并返回特定格式的注释列表。

这是因为 `image_processor` 期望注释的格式如下：`{'image_id': int, 'annotations': List[Dict]}`，其中每个字典是一个 COCO 对象注释。

```python
def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations
```

最后，我们将图像和注释的转换结合起来，对整个数据集批量进行转换。

以下是执行此操作的最终代码：

```python
# 批量转换

def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["id"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")
```

最后，你只需将此预处理函数应用于整个数据集。你可以通过使用 HuggingFace 🤗 的 [Datasets with transform](https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.with_transform) 方法来实现。

```python
# 对训练集和测试集进行转换

train_dataset_transformed = train_dataset.with_transform(transform_aug_ann)
test_dataset_transformed = test_dataset.with_transform(transform_aug_ann)
```

现在，让我们看看转换后的训练集样本是什么样的：

```python
train_dataset_transformed[0]
```

这将返回一个张量字典。我们主要需要的是代表图像的 `pixel_values`、作为注意力掩码的 `pixel_mask` 和标签 `labels`。以下是一个数据点的示例：

```
{'pixel_values': tensor([[[-0.1657, -0.1657, -0.1657,  ..., -0.3369, -0.4739, -0.5767],
          [-0.1657, -0.1657, -0.1657,  ..., -0.3369, -0.4739, -0.5767],
          [-0.1657, -0.1657, -0.1828,  ..., -0.3541, -0.4911, -0.5938],
          ...,
          [-0.4911, -0.5596, -0.6623,  ..., -0.7137, -0.7650, -0.7993],
          [-0.4911, -0.5596, -0.6794,  ..., -0.7308, -0.7993, -0.8335],
          [-0.4911, -0.5596, -0.6794,  ..., -0.7479, -0.8164, -0.8507]],
 
         [[-0.0924, -0.0924, -0.0924,  ...,  0.0651, -0.0749, -0.1800],
          [-0.0924, -0.0924, -0.0924,  ...,  0.0651, -0.0924, -0.2150],
          [-0.0924, -0.0924, -0.1099,  ...,  0.0476, -0.1275, -0.2500],
          ...,
          [-0.0924, -0.1800, -0.3200,  ..., -0.4426, -0.4951, -0.5301],
          [-0.0924, -0.1800, -0.3200,  ..., -0.4601, -0.5126, -0.5651],
          [-0.0924, -0.1800, -0.3200,  ..., -0.4601, -0.5301, -0.5826]],
 
         [[ 0.1999,  0.1999,  0.1999,  ...,  0.6705,  0.5136,  0.4091],
          [ 0.1999,  0.1999,  0.1999,  ...,  0.6531,  0.4962,  0.3916],
          [ 0.1999,  0.1999,  0.1825,  ...,  0.6356,  0.4614,  0.3568],
          ...,
          [ 0.4788,  0.3916,  0.2696,  ...,  0.1825,  0.1302,  0.0953],
          [ 0.4788,  0.3916,  0.2696,  ...,  0.1651,  0.0953,  0.0605],
          [ 0.4788,  0.3916,  0.2696,  ...,  0.1476,  0.0779,  0.0431]]]),
 'pixel_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
         [1, 1, 1,  ..., 1, 1, 1],
         [1, 1, 1,  ..., 1, 1, 1],
         ...,
         [1, 1, 1,  ..., 1, 1, 1],
         [1, 1, 1,  ..., 1, 1, 1],
         [1, 1, 1,  ..., 1, 1, 1]]),
 'labels': {'size': tensor([800, 800]), 'image_id': tensor([1]), 'class_labels': tensor([1, 1]), 'boxes': tensor([[0.5920, 0.3027, 0.1040, 0.1573],
         [0.7550, 0.4240, 0.0460, 0.0800]]), 'area': tensor([8522.2217, 1916.6666]), 'iscrowd': tensor([0, 0]), 'orig_size': tensor([480, 480])}}
```

我们快完成了 🚀。作为最后的预处理步骤，我们需要编写一个自定义的 `collate_fn`。那么，什么是 `collate_fn`？

`collate_fn` 负责将数据集中的一组样本转换为适合模型输入格式的批次。

一般来说，`DataCollator` 通常执行填充、截断等任务。在自定义 collate 函数中，我们通常定义如何将数据分组为批次，或简单地说，如何表示每个批次。

数据整合器主要是将数据组合在一起并进行预处理。让我们来编写我们的 collate 函数。

```python
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
```

## 训练 DETR 模型

目前，所有繁重的工作已经完成了。现在，剩下的就是一步一步组装每个部分。让我们开始吧！

训练过程包含以下步骤：

1. 使用与预处理相同的检查点，通过 [AutoModelForObjectDetection](https://huggingface.co/docs/transformers/v4.36.0/en/model_doc/auto#transformers.AutoModelForObjectDetection) 加载基础（预训练）模型。

2. 在 [TrainingArguments](https://huggingface.co/docs/transformers/v4.36.0/en/main_classes/trainer#transformers.TrainingArguments) 中定义所有超参数和附加参数。

3. 将训练参数与模型、数据集和图像一起传递到 [HuggingFace Trainer](https://huggingface.co/docs/transformers/v4.36.0/en/main_classes/trainer#transformers.Trainer)。

4. 调用 `train()` 方法微调模型。

> 在加载与预处理相同的检查点时，请记得传入您之前从数据集元数据创建的 `label2id` 和 `id2label` 映射。此外，我们指定 `ignore_mismatched_sizes=True`，以替换现有的分类头部为新的分类头部。

```python
from transformers import AutoModelForObjectDetection

id2label = {0: "head", 1: "helmet", 2: "person"}
label2id = {v: k for k, v in id2label.items()}


model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
```

在继续之前，登录 Hugging Face Hub 以便在训练时实时上传模型。这样，您无需处理检查点并将其存储在某处。

```python
from huggingface_hub import notebook_login

notebook_login()
```

完成后，开始训练模型。首先定义训练参数，然后定义使用这些参数进行训练的 trainer 对象，如下所示：

```python
from transformers import TrainingArguments
from transformers import Trainer

# 定义训练参数

training_args = TrainingArguments(
    output_dir="detr-resnet-50-hardhat-finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    max_steps=1000,
    fp16=True,
    save_steps=10,
    logging_steps=30,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
)

# 定义 trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset_transformed,
    eval_dataset=test_dataset_transformed,
    tokenizer=image_processor,
)

trainer.train()
```

训练完成后，您可以删除模型，因为检查点已上传到 Hugging Face Hub。

```python
del model
torch.cuda.synchronize()
```

### 测试和推理

现在我们将尝试对新微调的模型进行推理。在本教程中，我们将针对以下图像进行测试：

![input-test-image](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/test_input_for_od.png)

我们首先编写一个非常简单的代码来进行新图像的目标检测推理。我们从单张图片的推理开始，随后我们将整合一切并将其制作为一个函数。

```python
import requests
from transformers import pipeline

# 下载样本图像

url = "https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/test-helmet-object-detection.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 创建目标检测 pipeline

obj_detector = pipeline(
    "object-detection", model="anindya64/detr-resnet-50-dc5-hardhat-finetuned"
)
results = obj_detector(train_dataset[0]["image"])

print(results)
```

现在让我们编写一个非常简单的函数来将结果绘制在图像上。我们从结果中获得分数、标签和相应的边界框坐标，这些将用于在图像中绘制。

```python
def plot_results(image, results, threshold=0.7):
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    for result in results:
        score = result["score"]
        label = result["label"]
        box = list(result["box"].values())
        if score > threshold:
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=1)
            draw.text((x, y), label, fill="white")
            draw.text(
                (x + 0.5, y - 0.5),
                text=str(score),
                fill="green" if score > 0.7 else "red",
            )
    return image
```

最后，使用该函数处理我们使用的相同测试图像。

```
results = obj_detector(image)
plot_results(image, results)
```

输出将如下所示：

![output-test-image-plot](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/test_output_for_od.png)

现在，让我们将所有内容组合成一个简单的函数。

```python
def predict(image, pipeline, threshold=0.7):
    results = pipeline(image)
    return plot_results(image, results, threshold)


# 让我们测试另一个测试图像

img = test_dataset[0]["image"]
predict(img, obj_detector)
```

让我们使用我们的推理函数对多个图像进行绘制。

```python 
from tqdm.auto import tqdm


def plot_images(dataset, indices):
    """
    绘制图像及其注释。
    """
    num_rows = len(indices) // 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    for i, idx in tqdm(enumerate(indices), total=len(indices)):
        row = i // num_cols
        col = i % num_cols

        # 绘制图像
        image = predict(dataset[idx]["image"], obj_detector)

        # 在相应的子图上显示图像
        axes[row, col].imshow(image)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


plot_images(test_dataset, range(6))
```
运行此函数将会给出如下输出：

![test-sample-output-plot](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/sample_od_test_set_inference_output.png)

这结果还不错。如果我们进一步微调，还可以改善结果。您可以在此找到该微调检查点 [here](hf-vision/detr-resnet-50-dc5-harhat-finetuned)。