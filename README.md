<div align="center">

# 中文对话0.1B小模型 mini-ChatGPT-Chinese-0.1B  

中文  | [English](./README_en.md)  

</div>


# 一、👋介绍 
当前的大型语言模型通常参数量庞大，在普通消费级电脑上进行推理速度较慢，更不用提从零开始训练一个模型了。本项目的目标是从0开始训练一个生成式语言模型，包括tokenizer训练、模型预训练、SFT指令微调、RLHF优化(DPO)等。 

mini-ChatGPT-Chinese为中文对话小模型，模型参数只有0.1B（算共享权重约105M）。 


- 公开所有预训练、SFT指令微调、DPO偏好优化数据集来源。
- 训练过程中支持在任意位置停止，及在任意位置继续训练。
- 预训练：整合为端到端的预训练。
- SFT微调。
- RLHF偏好优化：使用DPO进行全量偏好优化。(待更新~)

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
GPT模型（Generative Pre-Trained Transformer），详情见论文: [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)。

模型配置见[config.py](https://github.com/JiaxingSong718/mini-ChatGPT-Chinese/blob/main/config/config.py)，官方的`GPT-3 Small`：`decoder layer `为12层。 

模型参数：0.1B。词表大小：13000，仅包含中文和少量英文。

## 2.3 训练过程
硬件：
```bash
# 预训练及sft阶段：
CPU: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz
内存：112 GB
显卡：NVIDIA 50%A100(40GB) * 1
```
1. **tokenizer 训练**： 现有`tokenizer`训练库遇到大语料时存在OOM问题，故全量语料按照类似`BPE`的方法根据词频合并、构造词库，运行耗时两天。

2. **预训练**：学习率为`1e-4`到`5e-3`的动态学习率，预训练时间为4天。

3. **prompt监督微调（SFT）**：使用`belle`指令训练数据集，学习率为`1e-7`到`5e-5`的动态学习率，微调时间1天。

4. **dpo直接偏好优化（RLHF）**：待更新

## 2.4 对话效果展示(还未做DPO)

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

存在问题：预训练数据集只有200万左右，模型参数也仅0.1B，不能涵盖所有领域，会有答非所问、废话生成器的情况。

# 三、📑使用说明

## 3.1 从克隆仓库代码开始

> [!CAUTION]
> 在预训练、SFT、RLFH阶段的`prompt`、`response`等字段，请务必加上`[EOS]`序列结束标记。   


### 3.1.1 克隆项目：
```bash
git clone https://github.com/JiaxingSong718/mini-ChatGPT-Chinese.git

cd mini-ChatGPT-Chinese
```
### 3.1.2 安装依赖 

本项目推荐使用`python 3.10`，过老的python版本可能不兼容所依赖的第三方库。  

pip安装：
```bash
pip install -r ./requirements.txt
```

## 3.2 Tokenizer训练  

**1.准备txt语料  **

本项目以wiki中文百科为主。获取中文wiki语料方法：中文Wiki下载地址：[zhwiki](https://dumps.wikimedia.org/zhwiki/)，下载`zhwiki-[存档日期]-pages-articles-multistream.xml.bz2`文件，大概2.7GB， 将下载的bz2文件转换为wiki.txt参考：[WikiExtractor](https://github.com/apertium/WikiExtractor)，再利用python的`OpenCC`库转换为简体中文，最后将得到的`wiki.simple.txt`放到项目根目录的`data`目录下即可。

训练tokenizer非常耗内存，如果你的语料非常大（合并后的`txt`文件超过2G），建议对语料按照类别、比例进行采样，以减少训练时间和内存消耗。

**2.训练tokenizer**

```
# 确保你的训练语料`txt`文件已经data目录下
cd Tokenizer
python train_tokenizer.py
```

训练得到13000词表的tokenizer的效果：
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

## 3.3 预训练 

1.预训练数据集示例

```
<|beginoftext|>气候变化指地球气候系统的长期变化，这种变化通常是由于人类活动所引起的大气中温室气体的排放而导致的。其可能的影响包括：极端天气现象的加剧，如暴雨、干旱、洪涝灾害；海平面上升，导致海岸线的改变；全球气温的上升，导致生态系统的崩溃和物种的灭绝；以及全球粮食安全和能源安全的问题等。<|endoftext|>
```

预训练前将数据集`encode`后存在`dataset.bin`中：

```
python build_dataset.py
```

运行预训练：

```bash
python train.py
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
