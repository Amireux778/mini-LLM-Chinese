# ----------------------------
# 1. 导入依赖库
# ----------------------------
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"  # 排除 GPU 3

import platform
import argparse  # 命令行参数解析
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist  # 分布式训练
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel  # DDP模型封装
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学习率调度
from torch.utils.data import DataLoader, DistributedSampler  # 分布式数据采样
from contextlib import nullcontext  # 上下文管理器（用于混合精度）

from transformers import AutoTokenizer  # HuggingFace分词器

# 自定义模块
from model.model import miniLLMChineseLM        # 模型定义
from model.LMConfig import LMConfig       # 模型配置
from model.dataset import PretrainDataset # 数据集类

warnings.filterwarnings('ignore')  # 忽略警告


# ----------------------------
# 2. 工具函数
# ----------------------------
def Logger(content):
    """分布式训练下仅主进程（Rank 0）打印日志"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """自定义学习率调度：余弦退火 + 小初始值"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# ----------------------------
# 3. 训练单个epoch
# ----------------------------
def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 损失函数（不自动求平均）
    start_time = time.time()
    
    # 遍历训练数据
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 数据移动到设备（GPU）
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 动态调整学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度训练上下文
        with ctx:
            res = model(X)  # 前向传播
            # 计算损失（带掩码）
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()  # 仅计算有效部分
            loss += res.aux_loss  # 附加损失（如MoE的负载平衡损失）
            loss = loss / args.accumulation_steps  # 梯度累积归一化

        # 反向传播（自动缩放损失）
        scaler.scale(loss).backward()

        # 梯度累积步骤控制
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪（防止梯度爆炸）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # 高效清空梯度

        # 日志记录
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复实际损失值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # WandB记录（仅主进程）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 模型保存（仅主进程）
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}_wiki6.pth'

            # 处理DDP模型的参数保存
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()  # 提取原始模型参数
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


# ----------------------------
# 4. 模型初始化
# ----------------------------
def init_model(lm_config):
    """初始化模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained('./model/wiki_tokenizer')
    model = miniLLMChineseLM(lm_config).to(args.device)
    # 打印可训练参数量
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


# ----------------------------
# 5. 分布式训练初始化
# ----------------------------
def init_distributed_mode():
    """初始化分布式训练环境"""
    if not ddp: return
    global ddp_local_rank, DEVICE

    # 使用NCCL后端（NVIDIA GPU专用）
    dist.init_process_group(backend="nccl")
    # ddp_rank = int(os.environ["RANK"])        # 全局进程编号

    # ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 当前节点内的GPU编号
    ddp_local_rank = dist.get_rank()  # 当前节点内的GPU编号
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)             # 绑定当前进程到指定GPU  

    # print("ddp_rank",ddp_rank)
    print("ddp_local_rank",ddp_local_rank)
    print("DEVICE",DEVICE)


# ----------------------------
# 6. 主程序入口
# ----------------------------
if __name__ == "__main__":
    # ----------------------------
    # 6.1 解析命令行参数
    # ----------------------------
    parser = argparse.ArgumentParser(description="mini-LLM-Chinese Pretraining")
    # 输出目录
    parser.add_argument("--out_dir", type=str, default="out")
    # 训练轮次（建议1轮快速验证，实际训练2-6轮）
    parser.add_argument("--epochs", type=int, default=6)
    # 批次大小（单GPU）
    parser.add_argument("--batch_size", type=int, default=32)
    # 初始学习率
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    # 训练设备
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # 混合精度类型（bfloat16/float16）
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # 是否启用WandB日志
    parser.add_argument("--use_wandb", action="store_true")
    # WandB项目名称
    parser.add_argument("--wandb_project", type=str, default="mini-LLM-Chinese-Pretrain")
    # 数据加载线程数
    parser.add_argument("--num_workers", type=int, default=4)
    # 是否启用分布式训练
    parser.add_argument("--ddp", action="store_true")
    # 梯度累积步数（模拟更大批次）
    parser.add_argument("--accumulation_steps", type=int, default=8)
    # 梯度裁剪阈值
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 学习率预热步数（本代码未使用）
    parser.add_argument("--warmup_iters", type=int, default=0)
    # 日志打印间隔
    parser.add_argument("--log_interval", type=int, default=100)
    # 模型保存间隔
    parser.add_argument("--save_interval", type=int, default=100)
    # 分布式训练本地Rank（自动填充，无需手动设置）
    parser.add_argument('--local_rank', type=int, default=-1)
    # 模型维度
    parser.add_argument('--dim', default=512, type=int)
    # 模型层数
    parser.add_argument('--n_layers', default=8, type=int)
    # 序列最大长度
    parser.add_argument('--max_seq_len', default=512, type=int)
    # 是否使用MoE（混合专家）
    parser.add_argument('--use_moe', default=False, type=bool)
    # 训练数据路径
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    # ----------------------------
    # 6.2 初始化配置
    # ----------------------------
    # 模型配置对象
    lm_config = LMConfig(
        dim=args.dim, 
        n_layers=args.n_layers, 
        max_seq_len=args.max_seq_len, 
        use_moe=args.use_moe
    )
    # 创建输出目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 计算每个迭代处理的token数（用于性能评估）
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)  # 固定随机种子
    
    # 设备类型（cuda/cpu）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    # WandB运行名称（包含关键参数）
    args.wandb_run_name = f"mini-LLM-Chinese-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    
    # 混合精度训练上下文（CPU无效果）
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ----------------------------
    # 6.3 分布式训练检测
    # ----------------------------
    # 通过环境变量判断是否处于DDP模式
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"  # 默认值


    # 初始化分布式训练
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # ----------------------------
    # 6.4 初始化日志和模型
    # ----------------------------
    # 初始化WandB（仅主进程）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
    
    print(wandb)

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)
    
    # ----------------------------
    # 6.5 准备数据集
    # ----------------------------
    train_ds = PretrainDataset(
        args.data_path, 
        tokenizer, 
        max_length=lm_config.max_seq_len
    )
    # 分布式采样器（保证不同GPU处理不同数据）
    train_sampler = DistributedSampler(train_ds) if ddp else None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,      # 锁页内存加速数据传输
        drop_last=False,       # 保留不完整批次
        shuffle=False,         # 采样器控制shuffle
        num_workers=args.num_workers,
        sampler=train_sampler  # 分布式采样器
    )

    # ----------------------------
    # 6.6 优化器和混合精度
    # ----------------------------
    # 梯度缩放器（混合精度训练）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ----------------------------
    # 6.7 分布式模型包装
    # ----------------------------
    if ddp:
        # 指定需要忽略同步的参数（如位置编码）
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # DDP封装模型
        model = DistributedDataParallel(
            model, 
            device_ids=[ddp_local_rank]  # 指定使用的GPU
        )

    # ----------------------------
    # 6.8 开始训练
    # ----------------------------
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
