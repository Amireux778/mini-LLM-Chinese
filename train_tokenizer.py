import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os

random.seed(42)


def train_tokenizer():
    # 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    data_path = 'dataset/pretrain_hq.jsonl'

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    special_tokens = ["<unk>", "<s>", "</s>"] #不同PLM原始论文和transformers库中数据的组织格式。其实，像Roberta，XLM等模型的中<s>, 
    # </s>是可以等价于Bert中的[CLS], [SEP]的，只不过不同作者的习惯不同。

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据
    texts = read_texts_from_jsonl(data_path)
    print(texts)

    files = [f",/data/wiki_simple.txt"]
    # 训练tokenizer
    tokenizer.train(files, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # 保存tokenizer
    tokenizer_dir = "model/wiki_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./model/wiki_tokenizer")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<unk>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<s>system\\n' + system_message + '</s>\\n' }}{% else %}{{ '<s>system\\n你是 mini-LLM-Chinese，是一个有用的人工智能助手。</s>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = Tokenizer.from_file("model/wiki_tokenizer")

    text = "Hello, y'all! How are you 😁 ? 这句话的中文是什么？"
    print(text)

    # 获取实际词汇表长度（包括特殊符号）
    # actual_vocab_size = len(tokenizer)
    # print('tokenizer实际词表长度：', actual_vocab_size)

    text_token=tokenizer.tokenize(text)
    print("编码后的token：",text_token)

    ids=tokenizer.encode(text, add_special_tokens=True)
    print("编码后token的ids：",ids)
    de=tokenizer.decode(ids, skip_special_tokens=True)
    print("ids解码后的文本：",de)


def main():
    # train_tokenizer()
    eval_tokenizer()


if __name__ == '__main__':
    main()
