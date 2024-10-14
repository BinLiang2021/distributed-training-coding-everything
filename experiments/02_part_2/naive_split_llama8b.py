import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 打印显存使用情况
def print_memory_usage(step):
    print(f"{step} - GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"{step} - GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# 定义模型的两个部分
class ModelPart1(nn.Module):
    def __init__(self, model):
        super(ModelPart1, self).__init__()
        children = list(model.children())
        self.model = nn.Sequential(*children[:len(children)//2])

    def forward(self, x):
        output = self.model(x)
        return output.last_hidden_state if hasattr(output, 'last_hidden_state') else output

class ModelPart2(nn.Module):
    def __init__(self, model):
        super(ModelPart2, self).__init__()
        children = list(model.children())
        self.model = nn.Sequential(*children[len(children)//2:])

    def forward(self, x):
        output = self.model(x)
        return output.logits if hasattr(output, 'logits') else output

# 训练函数
def train(rank, world_size, gpu_ids):
    setup(rank, world_size)

    # 获取当前进程的 GPU 设备编号
    gpu_id = gpu_ids[rank]
    torch.cuda.set_device(gpu_id)

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # 拆分模型
    model_part1 = ModelPart1(model).to(gpu_id)
    model_part2 = ModelPart2(model).to(gpu_id)

    # 包装模型部分以进行分布式数据并行
    model_part1 = DDP(model_part1, device_ids=[gpu_id])
    model_part2 = DDP(model_part2, device_ids=[gpu_id])

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model_part1.parameters()) + list(model_part2.parameters()), lr=0.01)

    # 生成一些随机数据
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(gpu_id)
    targets = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(gpu_id)

    # 训练循环
    for epoch in range(10):
        optimizer.zero_grad()

        # 前向传播
        #print_memory_usage("Before forward part1")
        outputs_part1 = model_part1(inputs)
        #print_memory_usage("After forward part1")

        #print_memory_usage("Before forward part2")
        outputs_part2 = model_part2(outputs_part1)
        #print_memory_usage("After forward part2")

        # 计算损失
        #print_memory_usage("Before loss computation")
        loss = criterion(outputs_part2.view(-1, outputs_part2.size(-1)), targets.view(-1))
        #print_memory_usage("After loss computation")

        # 反向传播
        #print_memory_usage("Before backward")
        loss.backward()
        #print_memory_usage("After backward")

        optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    gpu_ids = [6, 7]  # 使用 GPU 6 和 GPU 7
    world_size = len(gpu_ids)
    torch.multiprocessing.spawn(train, args=(world_size, gpu_ids), nprocs=world_size, join=True)