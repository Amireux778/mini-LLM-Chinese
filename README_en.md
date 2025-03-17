<div align="center">
# A Small Chat with 0.1B Chinese Language Model: mini-ChatGPT-Chinese-0.1B  

English | [Chinese](./README.md) 

</div>


# I.👋Introduction 
The parameters of modern large language models are often quite large, making inference slow on consumer-grade computers, let alone training a model from scratch. The goal of this project is to train a generative language model from 0, including tokenizer training, model pre-training, SFT instruction fine-tuning, RLHF optimization (DPO), etc.

**mini-ChatGPT-Chinese** is a small Chinese dialogue model with only 0.1B parameters (around 105M including shared weights).

- Openly disclose all sources of pre-training, SFT instruction fine-tuning, and DPO preference optimization datasets.
- Support stopping and resuming training at any point during the process.
- Pre-training: integrate into end-to-end pre-training.
- SFT fine-tuning.
- RLHF preference optimization: use DPO for full preference optimization (to be updated~).

# II.🛠️mini-ChatGPT-Chinese-0.1B model training process 

## 2.1 Pre-training Datasets

All datasets are from publicly available **single-turn dialogue** datasets. The main datasets include:

1. Community Q&A json version webtext2019zh - large-scale, high-quality dataset, see: [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus). A total of 4.1 million entries, approximately 2 million used for tokenizer training.
2. baike_qa2019 Baidu Encyclopedia Q&A, see: <https://aistudio.baidu.com/datasetdetail/107726>, a total of 1.4 million entries, approximately 200,000 selected.
3. Chinese medical field Q&A dataset, see: [Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data), around 790,000 entries.
4. Zhihu Q&A data, see: [Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL), around 1 million entries.
5. Instruction training data from Belle's open-source project, see: [train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN), around 500,000 entries.

The total dataset size is approximately 4.5 million entries: pre-training set: around 2 million, evaluation set: not yet set. SFT fine-tuning data is around 500,000, and the DPO optimization dataset is to be updated.

## 2.2 Model

GPT model (Generative Pre-Trained Transformer), for details, see the paper: [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165).

The model configuration can be found in [config.py](https://github.com/JiaxingSong718/mini-ChatGPT-Chinese/blob/main/config/config.py), official `GPT-3 Small`: `decoder layer` consists of 12 layers.

Model parameters: 0.1B. Vocabulary size: 13,000, including only Chinese and a small amount of English.

## 2.3 Training Process

Hardware:

```bash
# Pre-training and SFT phase:
CPU: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz
Memory: 112 GB
GPU: NVIDIA 50%A100(40GB) * 1
```

1. **Tokenizer Training**: Existing `tokenizer` training libraries encountered OOM issues with large datasets, so the entire corpus was merged and a vocabulary was constructed using a method similar to `BPE` based on word frequency. The process took two days.

2. **Pre-training**: Dynamic learning rate from `1e-4` to `5e-3`, pre-training took 4 days.

3. **Prompt Supervised Fine-tuning (SFT)**: Using the `belle` instruction training dataset, with a dynamic learning rate from `1e-7` to `5e-5`, fine-tuning took 1 day.

4. **DPO Direct Preference Optimization (RLHF)**: To be updated.

## 2.4 Conversation Effect Demonstration (DPO not yet implemented)

```
python client.py
```

```
User> 你好
Response>  <|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
你好
<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的吗？

User> 现在外面天气怎么样
Response<  <|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
现在外面天气怎么样
<|im_end|>
<|im_start|>assistant
抱歉，我无法回答实时天气问题，建议您查询当地的天气预报。

User> 中国的五大名湖
Response<  <|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
中国的五大名湖
<|im_end|>
<|im_start|>assistant
中国的五大名湖有太湖、洞庭湖、鄱阳湖、洪泽湖、麻湖。
```

There are issues: the pre-training dataset contains only around 2 million entries, and the model parameters are just 0.1B, which cannot cover all domains. This may result in instances of irrelevant answers and generate meaningless content.

# III. 📑 Usage Instructions

## 3.1 Starting from Cloning the Repository

> [!CAUTION]
> In the `prompt`, `response`, and other fields during the pre-training, SFT, and RLHF phases, be sure to include the `[EOS]` sequence end marker.

### 3.1.1 Clone the Project:

```bash
git clone https://github.com/JiaxingSong718/mini-ChatGPT-Chinese.git

cd mini-ChatGPT-Chinese
```

### 3.1.2 Install Dependencies

This project recommends using `python 3.10`; older Python versions may not be compatible with the required third-party libraries.

Install via pip:

```bash
pip install -r ./requirements.txt
```

## 3.2 Tokenizer Training

**1. Prepare the txt Corpus**

This project primarily uses the Chinese Wikipedia. To obtain the Chinese Wikipedia corpus, download the Chinese Wiki from the following link: [zhwiki](https://dumps.wikimedia.org/zhwiki/). Download the `zhwiki-[archive date]-pages-articles-multistream.xml.bz2` file, which is approximately 2.7GB. Convert the downloaded bz2 file to `wiki.txt` using [WikiExtractor](https://github.com/apertium/WikiExtractor), and then use Python's `OpenCC` library to convert it to Simplified Chinese. Finally, place the resulting `wiki.simple.txt` file in the `data` directory at the root of the project.

Training the tokenizer is memory-intensive. If your corpus is very large (the merged `txt` file exceeds 2GB), it is recommended to sample the corpus by category or proportion to reduce training time and memory consumption.

**2.Train the Tokenizer**

```
# Ensure your training corpus `txt` file is in the data directory
cd Tokenizer
python train_tokenizer.py
```

The result of training a tokenizer with a vocabulary size of 13,000:
```python
from Tokenizer import BPETokenizer

# 加载
tokenizer = BPETokenizer()
tokenizer.load('./tokenizer.bin')
idx,tokens,text = tokenizer.encode('<|im_start|>user\n给定一个英文句子，翻译成中文\n<|im_end|>')
print('encode: ',idx)
print('===================================================================')
print('token: ',tokens)
print('===================================================================')
print('text: ',text)
print('===================================================================')

s = tokenizer.decode(idx)
print('decode: ',s)
```

```
encode:  [13000, 13006, 10, 1026, 477, 569, 2947, 2194, 2589, 3578, 354, 2433, 10, 13001]
===================================================================
token:  [b'<|im_start|>', b'user', b'\n', b'\xe7\xbb\x99', b'\xe5\xae\x9a', b'\xe4\xb8\x80\xe4\xb8\xaa', b'\xe8\x8b\xb1\xe6\x96\x87', b'\xe5\x8f\xa5', b'\xe5\xad\x90\xef\xbc\x8c', b'\xe7\xbf\xbb\xe8\xaf\x91', b'\xe6\x88\x90', b'\xe4\xb8\xad\xe6\x96\x87', b'\n', b'<|im_end|>']
===================================================================
text:  ['<|im_start|>', 'user', '\n', '给', '定', '一个', '英文', '句', '子，', '翻译', '成', '中文', '\n', '<|im_end|>']
===================================================================
decode:  <|im_start|>user
给定一个英文句子，翻译成中文
<|im_end|>
```

## 3.3 Pre-training

Pre-training Dataset Example

```
<|beginoftext|>气候变化指地球气候系统的长期变化，这种变化通常是由于人类活动所引起的大气中温室气体的排放而导致的。其可能的影响包括：极端天气现象的加剧，如暴雨、干旱、洪涝灾害；海平面上升，导致海岸线的改变；全球气温的上升，导致生态系统的崩溃和物种的灭绝；以及全球粮食安全和能源安全的问题等。<|endoftext|>
```

Before pre-training, encode the dataset and save it in `dataset.bin`:

```
python build_dataset.py
```

Run the pre-training:

```bash
python train.py
```

## 3.4 SFT Fine-tuning

The SFT dataset comes from the contributions of [BELLE](https://github.com/LianjiaTech/BELLE), specifically from: [train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN), which has approximately 500,000 lines.

Example of SFT instruction fine-tuning dataset:

```json
{
    "prompt": "解释什么是“气候变化”，并概述其可能的影响。",
    "response": "气候变化指地球气候系统的长期变化，这种变化通常是由于人类活动所引起的大气中温室气体的排放而导致的。其可能的影响包括：极端天气现象的加剧，如暴雨、干旱、洪涝灾害；海平面上升，导致海岸线的改变；全球气温的上升，导致生态系统的崩溃和物种的灭绝；以及全球粮食安全和能源安全的问题等。"
}
```

```
训练语料(chatml格式)：
<|beginoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n解释什么是“气候变化”，并概述其可能的影响。<|im_end|>\n<|im_start|>assistant\n气候变化指地球气候系统的长期变化，这种变化通常是由于人类活动所引起的大气中温室气体的排放而导致的。其可能的影响包括：极端天气现象的加剧，如暴雨、干旱、洪涝灾害；海平面上升，导致海岸线的改变；全球气温的上升，导致生态系统的崩溃和物种的灭绝；以及全球粮食安全和能源安全的问题等。<|im_end|><|endoftext|>
```

Run the SFT fine-tuning:
``` bash
python sft.py
```

## 3.5 RLHF (Reinforcement Learning from Human Feedback) — To be updated ~

There are two main preference methods, PPO and DPO. Please search for papers and blogs for specific implementations. **This project uses the DPO fine-tuning method, which is comparatively memory-efficient.**

**DPO (Direct Preference Optimization) Fine-tuning** Based on the SFT model, fine-tuning can begin without training a reward model, simply by obtaining positive responses (chosen) and negative responses (rejected).

Example of DPO preference optimization dataset:

```json
{
    "prompt": "请介绍一下浙江大学",
    "chosen": "浙江大学是一所历史悠久、声誉卓著的高等学府，坐落于中国历史文化名城、风景旅游胜地杭州。",
    "rejected": "浙江大学是一所野鸡大学。"
}
```

Run preference optimization: To be updated ~

## 3.6 Inference

Ensure that the `model_save` and `Tokenizer` directories contain the following files:

```bash
mini-ChatGPT-Chinese
├─model_save
|  ├─SFT_GPT.pth
├─Tokenizer
|  └─tokenizer.bin
```

Run in the console:

```bash
python client.py
```

# IV.🎓Citation
If you find this project helpful, feel free to cite it.

```conf
@misc{mini-ChatGPT-Chinese,
    author={Jiaxing Song},
    title={mini-ChatGPT-Chinese-0.1B},
    year={2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/JiaxingSong718/mini-ChatGPT-Chinese}},
}
```

# V.🤔Other Matters
This project does not assume any risks or responsibilities arising from data security, public opinion risks, or any misguidance, misuse, dissemination, or improper use of the open-source model and code.
