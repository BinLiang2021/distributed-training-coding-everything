import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 定义一个简单的模型
class SimpleModelPart1(nn.Module):
    def __init__(self):
        super(SimpleModelPart1, self).__init__()
        self.linear = nn.Linear(10, 50)

    def forward(self, x):
        return self.linear(x)

class SimpleModelPart2(nn.Module):
    def __init__(self):
        super(SimpleModelPart2, self).__init__()
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        return self.linear(x)

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 训练函数
def train(rank, world_size, gpu_ids):
    setup(rank, world_size)

    # 获取当前进程的 GPU 设备编号
    gpu_id = gpu_ids[rank]

    # 创建模型的两个部分
    model_part1 = SimpleModelPart1().to(gpu_id)
    model_part2 = SimpleModelPart2().to(gpu_id)

    # 包装模型部分以进行分布式数据并行
    model_part1 = DDP(model_part1, device_ids=[gpu_id])
    model_part2 = DDP(model_part2, device_ids=[gpu_id])

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(list(model_part1.parameters()) + list(model_part2.parameters()), lr=0.01)

    # 生成一些随机数据
    inputs = torch.randn(32, 10).to(gpu_id)
    targets = torch.randn(32, 1).to(gpu_id)

    # 训练循环
    for epoch in range(10):
        optimizer.zero_grad()

        # 前向传播
        outputs_part1 = model_part1(inputs)
        outputs_part2 = model_part2(outputs_part1)

        # 计算损失
        loss = criterion(outputs_part2, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    gpu_ids = [6, 7]  # 使用 GPU 6 和 GPU 7
    world_size = len(gpu_ids)
    torch.multiprocessing.spawn(train, args=(world_size, gpu_ids), nprocs=world_size, join=True)
    