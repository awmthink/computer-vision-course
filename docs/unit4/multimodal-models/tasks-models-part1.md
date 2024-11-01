# 多模态任务与模型

在本节中，我们将简要介绍涉及图像和文本模态的多模态任务及其对应模型。在深入研究之前，我们先回顾一下前面章节中提到的“多模态”的含义。人类的世界是各种感官输入的交响曲，我们通过视觉、听觉、触觉等来感知和理解。这种多模态性让我们能够超越传统单模态 AI 模型的局限性。受人类认知的启发，多模态模型旨在通过整合来自多个来源（如文本、图像、音频，甚至是传感器数据）的信息来弥补这一差距。这种模态的融合带来了更全面和细致的世界理解，解锁了广泛的任务和应用场景。

## 任务示例
在研究特定模型之前，了解涉及图像和文本的多样化任务至关重要。这些任务包括但不限于：

- **视觉问答 (VQA) 和视觉推理**：想象一个机器能够看着一张图片，并理解你对图片的提问。视觉问答 (VQA) 正是这样！它训练计算机从图像中提取意义，并回答像“谁在开车？”这样的问题，而视觉推理则是其中的关键，使得机器能够超越简单的识别，推断关系、比较对象，并理解场景的上下文，以给出准确的答案。就像请一位侦探读取图片中的线索，只是更快、更精准！

- **文档视觉问答 (DocVQA)**：想象一台计算机能够理解文档的文本和布局（如地图或合同），并直接从图像中回答有关文档的问题。这就是文档视觉问答 (DocVQA)。它结合了计算机视觉来处理图像元素和自然语言处理来解读文本，使得机器能够像人类一样“阅读”并回答文档中的问题。可以把它看作是 AI 超级增强的文档搜索，解锁了那些图像中隐藏的信息。

- **图像字幕生成**：图像字幕生成弥合了视觉与语言之间的差距。它像侦探一样分析图像，提取细节、理解场景，并创建一两句话来描述——比如“宁静海面上的夕阳”，或者“秋千上笑着的孩子”，甚至是“熙熙攘攘的城市街道”。这是计算机视觉和语言的奇妙结合，让计算机能够用文字描述周围的世界，一张图片接着一张图片。

- **图像-文本检索**：图像-文本检索就像是图像和其描述的媒人。想象一下在图书馆中寻找一本特定的书，但不是浏览书名，而是通过封面图片或简要描述来找到它。这就像一个超强的搜索引擎，既能理解图像，也能理解文字，开启了许多有趣的应用，如图像搜索、自动字幕生成，甚至帮助视障人士通过文字描述“看见”世界。

- **视觉定位**：视觉定位就像将我们所见和所说的内容联系在一起。它的目标是理解语言如何指代图像中的特定部分，使 AI 模型能够根据自然语言描述来定位对象或区域。想象你问“水果碗里的红苹果在哪里？”然后 AI 即时在图像中高亮显示它——这就是视觉定位的应用！

- **文本生成图像**：想象一支能够解读你的文字并将其带入现实的神奇画笔！文本生成图像就像这样，它将你的书面描述转化为独特的图像。这是语言理解和图像创造的结合，让你的文字解锁一个视觉世界，从照片般逼真的风景到梦幻般的抽象，一切都源自你的语言之力。

## 视觉问答 (VQA) 和视觉推理
![VQA](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/vqa_visual_reasoning.png) *VQA 和视觉推理模型的输入（图像 + 文本）和输出（文本）示例 [[1]](#pretraining-paper)*

**视觉问答 (VQA)**
- **输入：** 图像-问题对（图像及关于该图像的一个问题）。
- **输出：** 在多项选择设置中：一个标签，表示在预定义的选项中对应正确答案的选项。
在开放式设置中：基于图像和问题的自由形式自然语言答案。
- **任务：** 回答关于图像的问题。（大多数 VQA 模型将其视为一个具有预定义答案的分类问题）。请参阅上方示例作为参考。

**视觉推理**
- **输入：** 根据特定的视觉推理任务而变化：
    - VQA 风格的任务：图像-问题对。
    - 匹配任务：图像和文本语句。
    - 蕴含任务：图像和文本对（可能包含多个语句）。
    - 子问题任务：图像和主要问题以及附加的感知相关子问题。
- **输出：** 根据任务的不同而变化：
    - VQA：图像问题的答案。
    - 匹配：判断文本是否真实地描述了图像（真/假）。
    - 蕴含：预测图像是否在语义上暗含文本。
    - 子问题：感知相关子问题的答案。
- **任务：** 执行多种图像上的推理任务。请参阅上方示例作为参考。

一般来说，视觉问答 (VQA) 和视觉推理任务都被视为*视觉问答 (VQA)* 任务。以下是一些流行的 VQA 模型：

- **BLIP-VQA**：这是 Salesforce AI 开发的一个大规模预训练模型，专用于视觉问答 (VQA) 任务。它采用了“自举式语言-图像预训练” (BLIP) 方法，通过利用噪声的网页数据和字幕生成，在各种视觉语言任务上达到了最先进的性能。你可以在 huggingface 中使用 BLIP，如下所示：
```python
from PIL import Image
from transformers import pipeline

vqa_pipeline = pipeline(
    "visual-question-answering", model="Salesforce/blip-vqa-capfilt-large"
)

image = Image.open("elephant.jpeg")
question = "Is there an elephant?"

vqa_pipeline(image, question, top_k=1)
```

- **Deplot**：这是一个单样本的视觉语言推理模型，专门训练用于将图表和图表翻译为文本摘要。这使其能够与 LLM 集成，以回答有关数据的复杂问题，即使是人类编写的全新查询。DePlot 通过标准化图表到表格的翻译任务并利用 Pix2Struct 架构，实现了图表问答的最新水平，仅凭一个示例和 LLM 提示即可超越之前的 SOTA。你可以在 huggingface 中使用 Deplot，如下所示：
```python
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

processor = Pix2StructProcessor.from_pretrained("google/deplot")
model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")

url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    images=image,
    text="Generate underlying data table of the figure below:",
    return_tensors="pt",
)
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```

- **VLIT**：这是一个视觉和语言 Transformer (ViLT) 模型，采用 Transformer 架构，无需卷积或区域监督，针对 VQAv2 数据集进行了微调，用于回答关于图像的自然语言问题。基础 ViLT 模型具有大规模架构 (B32 大小)，利用图像和文本的联合训练，在 VQA 等各种视觉语言任务中表现出色，具有竞争力的性能。你可以在 HuggingFace 中使用 VLIT，如下所示：
```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# 准备图像和问题
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# 准备输入
encoding = processor(image, text, return_tensors="pt")

# 前向传播
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
```
了解更多关于如何在 HuggingFace `transformers` 库中训练和使用 VQA 模型，请点击[此处](https://huggingface.co/docs/transformers/v4.36.1/tasks/visual_question_answering)。

## 文档视觉问答 (DocVQA)
![DocVQA](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/doc_vqa.jpg)
*Doc VQA 模型的输入 (图像 + 文本) 和输出 (文本) 示例。 [[2]](#doc-vqa-paper)*

- **输入：**
    - 文档图像：一个包含文本、布局和视觉元素的扫描或数字文档图像。
    - 关于文档的问题：一个以自然语言文本格式呈现的问题。

- **任务：**
    - 分析和理解：DocVQA 模型必须处理文档中的视觉和文本信息，以完全理解其内容。
    - 推理和推断：模型需要在视觉元素、文本和问题之间建立联系，以得出相关结论。
    - 生成自然语言答案：模型必须以自然语言文本格式生成清晰、简洁且准确的答案。参见上面的示例作为参考。

- **输出：**问题的答案：直接回答查询并准确反映文档中找到的信息的文本响应。

接下来，让我们看一下 HuggingFace 中一些流行的 DocVQA 模型：

- **LayoutLM**：这是一个预训练神经网络模型，通过联合分析文本及其布局来理解文档图像。与传统 NLP 模型不同，它会考虑字体大小、位置和邻近度等因素，以学习单词间的关系及其在文档上下文中的含义。使其在表单理解、收据分析和文档分类等任务中表现优异，成为从扫描文档中提取信息的强大工具。你可以在 HuggingFace 中使用 LayoutLM，如下所示：
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

question = "What is the purchase amount?"
image = Image.open("your-document.png")

pipe(image=image, question=question)

## [{'answer': '20,000$'}]
```

- **Donut：** 也被称为 OCR-free Document Understanding Transformer，是一种先进的图像处理模型，它绕过传统的光学字符识别（OCR）直接分析文档图像以理解其内容。它结合了视觉编码器（Swin Transformer）和文本解码器（BART），以提取信息并生成文本描述，擅长于文档分类、表单理解和视觉问答等任务。其独特的优势在于其“端到端”特性，避免了单独的 OCR 步骤带来的潜在错误，并通过高效的处理实现了令人印象深刻的准确性。您可以在 HuggingFace 中使用 Donut 模型，如下所示：
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa"
)

question = "What is the purchase amount?"
image = Image.open("your-document.png")

pipe(image=image, question=question)

## [{'answer': '20,000$'}]
```
- **Nougat：** 这是一种视觉 Transformer 模型，训练于数百万篇学术论文上，可以直接“读取”扫描的 PDF，并以结构化标记语言输出其内容，甚至能够理解复杂的元素如数学公式和表格。它绕过了传统的光学字符识别，实现了高精度并保持了语义完整，使存储在 PDF 中的科学知识更易于访问和使用。Nougat 使用与 Donut 相同的架构，即图像 Transformer 编码器和自回归文本 Transformer 解码器，将科学 PDF 转换为 markdown，以便更容易访问。您可以在 HuggingFace 中使用 Nougat 模型，如下所示：
```python
from huggingface_hub import hf_hub_download
import re
from PIL import Image

from transformers import NougatProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# 为模型准备 PDF 图像
filepath = hf_hub_download(
    repo_id="hf-internal-testing/fixtures_docvqa",
    filename="nougat_paper.png",
    repo_type="dataset",
)
image = Image.open(filepath)
pixel_values = processor(image, return_tensors="pt").pixel_values

# 生成转录（此处我们只生成 30 个 token）
outputs = model.generate(
    pixel_values.to(device),
    min_length=1,
    max_new_tokens=30,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
# 注意：我们在此处使用 repr 只是为了打印 \n 字符，您可以直接打印 sequence
print(repr(sequence))
```
了解更多关于如何在 HuggingFace 的 `transformers` 库中训练和使用 DocVQA 模型的内容，请参考[这里](https://huggingface.co/docs/transformers/tasks/document_question_answering)。

## 图像描述
![Image Captioning](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/image_captioning.png)
*图像描述模型的输入（图像）和输出（文本）的示例。[[1]](#pretraining-paper)*
- **输入：**
    - 图像：多种格式的图像（例如 JPEG，PNG）。
    - 预训练的图像特征提取器（可选）：一种预训练的神经网络，可从图像中提取有意义的特征，如卷积神经网络（CNN）。
- **输出：** 文本描述：准确描述输入图像内容的单句或段落，捕捉对象、动作、关系及整体上下文。参见上图示例。
- **任务：** 自动生成图像的自然语言描述。包括：(1) 理解图像的视觉内容（对象、动作、关系）。(2) 将该信息编码为有意义的表示。(3) 将此表示解码为连贯、语法正确且信息丰富的句子或短语。

接下来，让我们看看 HuggingFace 中一些流行的图像描述模型：
- **ViT-GPT2：** 这是一种用于生成图像描述的 PyTorch 模型，通过将 Vision Transformer (ViT) 用于视觉特征提取和 GPT-2 用于文本生成构建而成。在 COCO 数据集上训练，利用 ViT 编码丰富的图像细节以及 GPT-2 生成流畅语言的能力，以创建准确和描述性的文本。此开源模型为图像理解和描述任务提供了有效的解决方案。您可以在 HuggingFace 中使用 **ViT-GPT2**，如下所示：
```python
from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")

# [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]
```
- **BLIP-Image-Captioning：** 这是基于 BLIP 的一种先进图像描述模型，BLIP 是一种在清洁和噪声的网络数据上预训练的框架，适用于统一的视觉语言理解和生成。它利用自举过程来过滤噪声描述，在图像描述、图像-文本检索和视觉问答等任务中表现更佳。此大型版本以 ViT-L 主干为基础，擅长从图像生成准确且详细的描述。您可以在 HuggingFace 中使用 BLIP 图像描述模型，如下所示：
```python
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# 条件图像描述
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# 无条件图像描述
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
```
- **git-base：** `microsoft/git-base` 是 GIT（GenerativeImage2Text）模型的基础版，一种 Transformer 解码器，旨在生成图像的文本描述。它同时接收图像 token 和文本 token 作为输入，并基于图像和先前的文本预测下一个文本 token。它适用于图像和视频描述等任务。已微调的版本如 `microsoft/git-base-coco` 和 `microsoft/git-base-textcaps` 针对特定数据集，而基础模型则为进一步定制提供了起点。您可以在 HuggingFace 中使用 git-base 模型，如下所示：
```python
from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
```

了解更多关于如何在 HuggingFace 的 `transformers` 库中训练和使用图像描述模型的内容，请参考[这里](https://huggingface.co/docs/transformers/v4.36.1/en/tasks/image_captioning)。

## 图像-文本检索
![Image-Text Retrieval](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/image_text_retrieval.png)
*文本到图像检索的输入（文本查询）和输出（图像）示例。[[1]](#pretraining-paper)*

- **输入：**
    - 图像：各种格式的图像（如 JPEG、PNG）。
    - 文本：自然语言文本，通常以标题、描述或与图像相关的关键词形式出现。
- **输出：**
    - 相关图像：当提供文本查询时，系统返回与文本最相关的图像的排序列表。
    - 相关文本：当提供图像查询时，系统返回最能描述该图像的文本描述或标题的排序列表。
- **任务：**
    - 图像到文本检索：给定图像作为输入，检索准确描述其内容的文本描述或标题。
    - 文本到图像检索：给定文本查询，检索与文本中提到的概念和实体在视觉上相符的图像。

CLIP 是图像-文本检索中最流行的模型之一。

- **CLIP（对比语言-图像预训练）：** 通过共享的嵌入空间在图像-文本检索中表现出色。它通过对比学习在大规模的图像和文本数据集上预训练，使模型能够将多种概念映射到一个公共空间。在图像-文本检索中，可以通过将图像和文本编码到共享的嵌入空间中来应用 CLIP，并通过其各自嵌入的接近度来测量图像与文本查询之间的相似性。该模型的多功能性在于其无需任务特定的微调即可掌握语义关系，使其高效适用于从基于内容的图像检索到自然语言图像查询的解释等应用。你可以在 HuggingFace 中如下使用 CLIP 进行图像-文本检索：
```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 这是图像-文本相似度得分
probs = logits_per_image.softmax(
    dim=1
)  # 我们可以取 softmax 得到标签概率
```
在 HuggingFace [这里](https://huggingface.co/docs/transformers/model_doc/clip#resources)了解如何使用 CLIP 进行图像-文本检索。

## 视觉定位
![Visual Grounding](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/visual_grounding.jpg)
*输入（图像 + 文本）和输出（边界框）的示例。(a) 短语定位 (b) 表达理解。[[1]](#pretraining-paper)*

- **输入：**
    - 图像：场景或对象的视觉表示。
    - 自然语言查询：指向图像中特定部分的文本描述或问题。

- **输出：** 边界框或分割掩码：图像中对应于查询描述的对象或区域的空间区域，通常以坐标或高亮区域表示。
- **任务：** 定位相关对象或区域：模型必须正确识别图像中与查询匹配的部分，这涉及到对图像的视觉内容和查询的语言含义的理解。

以下是一些在 HuggingFace 上流行的视觉定位（对象检测）模型。

- **OWL-ViT：** OWL-ViT（用于开放世界定位的视觉 Transformer）是一种基于标准视觉 Transformer 架构并在大规模图像-文本对上训练的强大对象检测模型。它擅长“开放词汇”检测，这意味着它可以根据文本描述识别未出现在训练数据中的对象。通过对比预训练和微调，OWL-ViT 在零样本（基于文本引导）和一次样本（基于图像引导）检测任务中表现出色，使其成为图像中灵活搜索和识别的多功能工具。你可以如下在 HuggingFace 上使用 OWL-ViT：
```python
import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# 目标图像尺寸（高度，宽度）以重新缩放框预测 [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# 将输出（边界框和类别 logits）转换为 COCO API 格式
results = processor.post_process_object_detection(
    outputs=outputs, threshold=0.1, target_sizes=target_sizes
)

i = 0  # 获取第一个图像对应文本查询的预测结果
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# 打印检测到的对象和重新缩放的边框坐标
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
    )
```
- **Grounding DINO[[3]](#grounding-dino)：** 它结合了基于 Transformer 的对象检测器（DINO）和“基础预训练”来创建一个最先进的零样本对象检测模型。这意味着即使它们不在训练数据中，它也能识别对象，这得益于它理解图像和人类语言输入（如类别名称或描述）的能力。其架构结合了文本和图像主干、特征增强器、语言引导的查询选择和跨模态解码器，在 COCO 和 LVIS 等基准测试中取得了出色的成绩。Grounding DINO 从视觉信息中获取，将其与文本描述关联，并使用这种理解来精确定位全新类别中的对象。
你可以在 Google Colab [这里](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb)尝试 Grounding DINO 模型。

## 文本到图像生成
![Text-Image Generation](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/text_image_generation.png)
*自回归和扩散模型用于文本到图像生成的示例。[[1]](#pretraining-paper)*

- **自回归模型：** 这些模型将任务视为将文本描述转换为图像标记序列，类似于语言模型生成句子。类似拼图，这些标记由像 VQ-VAE 这样的图像标记器创建，代表图像的基本特征。该模型使用编码器-解码器架构：编码器从文本提示中提取信息，解码器在该信息的指导下逐个预测图像标记，逐渐生成最终的图像。这种方法允许高控制和细节，但在处理长而复杂的提示方面存在挑战，且速度可能比扩散模型更慢。生成过程如上图（a）所示。

- **稳定扩散模型：** 稳定扩散模型使用“潜在扩散”技术，通过逐步对噪声去噪生成图像，指导该过程的是文本提示和冻结的 CLIP 文本编码器。其轻量级架构使用 UNet 主干和 CLIP 编码器，使 GPU 驱动的图像生成成为可能，同时其潜在聚焦降低了内存消耗。这种独特的设置支持多样化的艺术表达，将文本输入转化为逼真且富有想象力的视觉效果。生成过程如上图（b）所示。

现在，让我们看看如何在 HuggingFace 中使用文本-图像生成模型。

安装 `diffusers` 库：
```bash
pip install diffusers --upgrade
```

此外，确保安装 transformers、safetensors、accelerate 以及 invisible watermark：
```bash


pip install invisible_watermark transformers accelerate safetensors
```
要仅使用基础模型，可以运行以下代码：
```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")

prompt = "An astronaut riding a unicorn"

images = pipe(prompt=prompt).images[0]
```

要了解更多关于文本-图像生成模型的信息，可以参考 HuggingFace 的 [Diffusers 课程](https://huggingface.co/docs/diffusers/training/overview)。

现在你已经了解了一些涉及图像和文本模态的流行任务和模型，但你可能会好奇如何训练或微调上述任务。接下来，让我们简要了解视觉-语言模型的训练。

## 视觉-语言预训练模型概览

![VLP Framework](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/multimodal_fusion_text_vision/vlp_framework.png)
*基于 Transformer 的视觉-语言模型的通用框架。[[1]](#pretraining-paper)*

给定一个图像-文本对，VL 模型首先通过文本编码器和视觉编码器分别提取文本特征和视觉特征，然后将这些特征输入到多模态融合模块中，以生成跨模态表示，这些表示可以在生成最终输出之前可选地输入到解码器中。上述图显示了这一通用框架。在许多情况下，图像/文本主干、多模态融合模块和解码器之间没有明确的边界。

恭喜你！你已经到达最后，现在进入下一部分，了解更多视觉-语言预训练模型。

## 参考文献

1. [视觉-语言预训练：基础、最新进展及未来趋势](https://arxiv.org/abs/2210.09263)<a id="pretraining-paper"></a>
2. [文档集合视觉问答](https://arxiv.org/abs/2104.14336)<a id="doc-vqa-paper"></a>
3. [Grounding DINO：结合 DINO 和基础预训练进行开放集对象检测](https://arxiv.org/abs/2303.05499)<a id="grounding-dino"></a>