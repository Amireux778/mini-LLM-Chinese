import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer    # 分词器（如HuggingFace Tokenizer）
        self.max_length = max_length  # 序列最大长度
        self.samples = self.load_data(data_path)  # 加载原始数据

    def load_data(self, path):
        """从jsonl文件加载预训练语料"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())  # 解析每行JSON数据
                samples.append(data)
        print(f"加载预训练样本数：{len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]  #{"text":" "}
        # 构建输入文本（添加BOS/EOS标记）
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"   #" "
        # 分词与编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',     # 填充到max_length
            truncation=True,         # 超长截断
            return_tensors='pt'       # 返回PyTorch张量
        )
        input_ids = encoding.input_ids.squeeze()  # 移除批次维度  input_ids:内容token对应的词表序号

        # 生成损失掩码（忽略填充部分的损失）
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  #[True...,False...],,填充部分为False，具体内容部分为True

        # 构造训练数据（预测下一个token）
        X = input_ids[:-1]  # 输入序列（去掉最后一个token）
        Y = input_ids[1:]   # 目标序列（去掉第一个token）
        # print("x",X)
        # print("y",Y)

        loss_mask = loss_mask[1:]  # 对齐Y的位置
        #print("lose_mask",loss_mask)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 提取角色标记的ID（用于动态损失掩码）

        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids  # 回答开始标记
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids         # 回答结束标记

    def load_data(self, path):
        """加载对话格式的微调数据"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())  # 解析对话数据
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """将对话转换为ChatML格式的提示"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'  # 交替角色
            messages.append({"role": role, "content": turn['content']})
        # print("*********************")
        # print(messages)
        # print("*********************")
        # 应用模板生成结构化文本（如添加<|im_start|>等标记）
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,              # 仅生成文本不进行分词
            add_generation_prompt=False  # 不添加生成提示
        )

    def _generate_loss_mask(self, input_ids):
        """生成动态损失掩码：仅在assistant回答部分计算损失"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 检测回答开始标记（如<assistant>）
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)  # 回答内容起始位置
                end = start
                # 寻找回答结束标记（如</s>）
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 标记需要计算损失的区域（回答内容+结束符）
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    #自己加的
    def __len__(self):
            return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # 生成对话提示文本
        prompt = self._create_chat_prompt(sample['conversations'])

        # 编码并填充/截断
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # 生成动态掩码（仅计算模型回答部分的损失）
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据（预测下一个token）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        return X, Y, loss_mask
    


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id else 0  # 填充ID
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        # 加载DPO偏好数据（包含chosen和rejected样本）
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # 处理优选（chosen）和拒绝（rejected）样本
        chosen = item['chosen']    # 优选对话历史
        rejected = item['rejected']# 拒绝对话历史
        
        # 生成对话提示文本
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False)
        
        # 编码优选和拒绝样本
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 生成损失掩码
        chosen_loss_mask = self._generate_loss_mask(chosen_encoding['input_ids'])
        rejected_loss_mask = self._generate_loss_mask(rejected_encoding['input_ids'])

        # 转换为张量
        x_chosen = torch.tensor(chosen_encoding['input_ids'][:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_encoding['input_ids'][1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_encoding['input_ids'][:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_encoding['input_ids'][1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,        # 优选输入序列
            'y_chosen': y_chosen,        # 优选目标序列
            'mask_chosen': mask_chosen,  # 优选损失掩码
            'x_rejected': x_rejected,    # 拒绝输入序列
            'y_rejected': y_rejected,     # 拒绝目标序列
            'mask_rejected': mask_rejected# 拒绝损失掩码
        }

    def _generate_loss_mask(self, input_ids):
        """与SFT相同的掩码生成逻辑"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    


if __name__ == "__main__":
    pass
