import math
import struct
import inspect
import time

from .LMConfig import LMConfig  # 导入模型配置类
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel  # Hugging Face 的预训练模型基类
from transformers.modeling_outputs import CausalLMOutputWithPast  # 语言模型输出类


# RMSNorm 归一化层
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps  # 防止除零的小常数
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x):
        # RMSNorm 归一化：计算输入的均方根并进行归一化
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


# 预计算旋转位置编码
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    # 计算频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # 位置索引
    freqs = torch.outer(t, freqs).float()  # 外积计算频率矩阵
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 转换为复数形式
    return pos_cis


# 应用旋转位置编码
def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        # 调整 pos_cis 的形状以匹配输入 x
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # 将输入转换为复数形式
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    # 应用旋转位置编码
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 重复 Key 和 Value
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复 Key 和 Value 以匹配 Query 的数量"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# 注意力机制
class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)  # Query 投影
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # Key 投影
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # Value 投影
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # 输出投影
        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力 Dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差 Dropout
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 因果掩码
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, past_key_value=None, use_cache=False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)  # 应用旋转位置编码
        # KV Cache 实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash and seq_len != 1:
            # 使用 Flash Attention
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            # 标准注意力计算
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]  # 应用因果掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))  # 输出投影
        return output, past_kv


# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # 第一层
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)  # 第二层
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # 第三层
        self.dropout = nn.Dropout(config.dropout)  # Dropout

    def forward(self, x):
        # 使用 SiLU 激活函数
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# MoE 门控机制
class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个 token 选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 专家数量
        self.scoring_func = config.scoring_func  # 评分函数
        self.alpha = config.aux_loss_alpha  # 辅助损失权重
        self.seq_aux = config.seq_aux  # 是否使用序列级辅助损失
        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化 top-k 概率
        self.gating_dim = config.dim  # 门控维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 门控权重
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 初始化门控权重
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)  # 计算 logits
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 使用 softmax 计算分数
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # 选择 top-k 专家

        if self.top_k > 1 and self.norm_topk_prob:
            # 归一化 top-k 概率
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            # 计算辅助损失
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


# MoE 前馈网络
class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)  # 初始化多个专家
        ])
        self.gate = MoEGate(config)  # 门控机制
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)  # 共享专家

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)  # 添加共享专家输出
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 推理模式下的 MoE 计算
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache


# mini-LLM-Chinese 模型块
class miniLLMChineseBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)  # 注意力机制
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)  # 注意力归一化
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)  # 前馈网络归一化
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)  # 前馈网络

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn  # 残差连接
        out = h + self.feed_forward(self.ffn_norm(h))  # 前馈网络 + 残差连接
        return out, past_kv


# mini-LLM-Chinese 语言模型
class miniLLMChineseLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)  # 词嵌入
        self.dropout = nn.Dropout(params.dropout)  # Dropout
        self.layers = nn.ModuleList([MminiLLMChineseBlock(l, params) for l in range(self.n_layers)])  # 模型层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # 最终归一化
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # 输出层
        self.tok_embeddings.weight = self.output.weight  # 共享权重
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)
        h = self.dropout(self.tok_embeddings(input_ids))  # 词嵌入
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]  # 位置编码
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))  # 输出 logits
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        # 流式生成逻辑
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break