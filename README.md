<div align="center">

# 中文对话0.02B小模型 mini-llm-0.02B  

中文  | [English](./README_en.md)  

</div>


# 一、👋介绍 
本项目的目标是从0开始训练一个生成式语言模型，包括tokenizer训练、模型预训练、SFT指令微调、RLHF优化(DPO)等。 

mini-llm-0.02B 为中文对话小模型，模型参数只有0.02B（算共享权重约25.83M）。 


- 公开所有预训练、SFT指令微调、DPO偏好优化数据集来源。
- 训练Tokenizer
- 预训练：整合为端到端的预训练。
- SFT微调。
- RLHF偏好优化：使用DPO进行全量偏好优化。

# 二、🛠️mini-ChatGPT-Chinese-0.1B模型训练过程 

## 2.1 预训练数据集
所有数据集均来自互联网公开的**单轮对话**数据集。主要数据集包括： 

1. 社区问答json版webtext2019zh-大规模高质量数据集，见：[nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)。共410万，选取大约200万条用于tokenizer训练。
2. baike_qa2019百科类问答，见：<https://aistudio.baidu.com/datasetdetail/107726>，共140万，选取大约20万条。
3. 中国医药领域问答数据集，见：[Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)，共79万条左右。
5. 知乎问答数据，见：[Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL)，共100万条左右。
6. belle开源的指令训练数据，见：[train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)，共50万条左右。

数据集总数量250万左右：预训练集：200万左右，评估集：还未设置。 SFT微调数据大概50万和DPO优化数据集待更新。

## 2.2 模型
（待补充）

模型参数：0.02B。词表大小：6400，仅包含中文和少量英文。

## 2.3 训练过程
硬件：
```bash
# 预训练及sft阶段：
CPU: AMD EPYC 7763 64-Core Processor
内存：512 GB
显卡：NVIDIA GeForce RTX 4090(24G) * 8
```
1. **tokenizer 训练**： 训练时间40分钟左右

2. **预训练**：（待补充）

3. **prompt监督微调（SFT）**：（待补充）

4. **dpo直接偏好优化（RLHF）**：待更新

存在问题：预训练数据集只有200万左右，模型参数也仅0.02B，不能涵盖所有领域，会有答非所问、废话生成器的情况。


# 三、📑使用说明


## 3.2 Tokenizer训练  

**1.准备txt语料  **

本项目以wiki中文百科为主。获取中文wiki语料方法：中文Wiki下载地址：[zhwiki](https://dumps.wikimedia.org/zhwiki/)，下载`zhwiki-[存档日期]-pages-articles-multistream.xml.bz2`文件，大概3GB， 将下载的bz2文件转换为wiki.txt参考：[WikiExtractor](https://github.com/apertium/WikiExtractor)，再利用python的`OpenCC`库转换为简体中文，最后将得到的`wiki.simple.txt`放到项目根目录的`data`目录下即可。

训练tokenizer非常耗内存，如果你的语料非常大（合并后的`txt`文件超过2G），建议对语料按照类别、比例进行采样，以减少训练时间和内存消耗。

**2.训练tokenizer**

```
# 确保你的训练语料`txt`文件已经data目录下
cd Tokenizer
python train_tokenizer.py
```
**3.测试训练好的tokenizer**
训练得到6400词表的tokenizer的效果：
```python
from transformers import AutoTokenizer

# 加载预训练的tokenizer
# tokenizer = AutoTokenizer.from_pretrained("./minimind_tokenizer")      #pretrain_hq.jsonl 训练的
tokenizer = AutoTokenizer.from_pretrained("./wiki_tokenizer")            # wiki_simple.txt 训练的
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")         # hf上的

text = "Hello, y'all! How are you 😁 ? 这句话的中文是什么？"

# 获取实际词汇表长度（包括特殊符号）
actual_vocab_size = len(tokenizer)
print('tokenizer实际词表长度：', actual_vocab_size)

text_token=tokenizer.tokenize(text)
print("编码后的token：",text_token)

ids=tokenizer.encode(text, add_special_tokens=True)
print("编码后token的ids：",ids)

de=tokenizer.decode(ids, skip_special_tokens=True)
print("ids解码后的文本：",de)
```

```
tokenizer实际词表长度： 6400
编码后的token： ['H', 'ell', 'o', ',', 'Ġ', 'y', "'", 'all', '!', 'ĠH', 'ow', 'Ġ', 'are', 'Ġ', 'y', 'ou', 'Ġ', 'ð', 'Ł', 'ĺ', 'ģ', 'Ġ', '?', 'Ġ', 'è¿Ļ', 'åı¥', 'è¯Ŀ', 'çļĦ', 'ä¸Ńæĸĩ', 'æĺ¯', 'ä»Ģä¹Ī', 'ï¼Ł']
编码后token的ids： [42, 4188, 81, 14, 223, 91, 9, 4777, 3, 2231, 2349, 223, 6271, 223, 91, 1738, 223, 175, 256, 249, 226, 223, 33, 223, 458, 3253, 1408, 265, 3030, 305, 4140, 3287]
ids解码后的文本： Hello, y'all! How are you 😁 ? 这句话的中文是什么？
```

## 3.3 预训练 
1.预训练数据集为jsonl文件，每行是一个独立的有效 JSON 对象。
文件地址：（待补充）

2.预训练数据集示例
```
{"text": "<s>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。</s> <s>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？</s> <s>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。</s> <s>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。</s> <s>非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。</s> <s>帮我想一个有趣的标题。这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。</s> <s>回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。</s> <s>识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。</s>"}
```
3.预训练实现：（待补充）
    目前已实现完整训练流程
    后续准备实现混合精度训练和torchrun分布式训练
4.运行预训练：
```
python train_pretrain.py
```
5.预训练结果展示：（待补充）
```
***故事续写***
👶输入: 在一个遥远的星球上，有一种神奇的生物
🤖️输出: 叫做“神龙”。神龙有一个巨大的龙身，闪闪发光，但又有一个神奇的东西——“神龙”。神龙之所以能够发挥最神奇的能力，是能够让人类能够看到、发现、感知和探索世界的力量。神龙之所以能够发挥最神奇的能力，是能够让人类看到、发现、感知、探索世界的力量。
***对话生成***
👶输入: 用户：今天的天气怎么样？助手：
🤖️输出: 是的，天气很好，记得带伞。
***事实问答***
👶输入: 问题：太阳系中最大的行星是哪个？答案：
🤖️输出: 木星。太阳系中最大的行星是木星。木星是太阳系中最大的行星，它的直径是地球的11倍。木星是太阳系中最大的行星，其质量是其他行星的总和的318倍。
```



## 3.4 SFT微调 
SFT数据集来自[BELLE](https://github.com/LianjiaTech/BELLE)大佬的贡献，SFT数据集为：[train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)，约50万行。

sft指令微调数据集示例：

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

运行SFT微调：
``` bash
python sft.py
```

## 3.5 RLHF（强化学习人类反馈优化方法）——待更新~

偏好方法主要有两种，分别是PPO和DPO，具体实现请自行搜索论文及博客，**本项目采用DPO微调方法，比较节省显存**。 

**DPO（直接偏好优化，Direct Preference Optimization）微调**
在获得SFT模型的基础上，无需训练奖励模型，取得正向回答（chosen）和负向回答（rejected）即可开始微调。

DPO偏好优化数据集示例：
```json
{
    "prompt": "请介绍一下浙江大学",
    "chosen": "浙江大学是一所历史悠久、声誉卓著的高等学府，坐落于中国历史文化名城、风景旅游胜地杭州。",
    "rejected": "浙江大学是一所野鸡大学。"
}
```

运行偏好优化：待更新~

## 3.6 推理 
确保`model_save`和`Tokenizer`目录下有以下文件：
```bash
mini-ChatGPT-Chinese
├─model_save
|  ├─SFT_GPT.pth
├─Tokenizer
|  └─tokenizer.bin
```

控制台运行：

```bash
python client.py
```

# 四、🎓引用
如果你觉得本项目对你有所帮助，欢迎引用。
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

# 五、🤔其他事项
本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。
