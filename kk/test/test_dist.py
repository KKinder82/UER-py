import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed as dist_data
import torch.nn.parallel.distributed as dist_nn
import torch.utils.data as data

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.net = nn.Linear(5, 1)

    def forward(self, x):
        return self.net(x)

def test():
    x = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    y = torch.ones(10, dtype=torch.float32)
    dataset = zip(x, y)
    sampler = dist_data.DistributedSampler(dataset, num_replicas=2, rank=0)
    loader = data.DataLoader(dataset, batch_size=2, sampler=sampler)
    model = TestModel()
    dist.init_process_group(backend="nccl", init_method="env://", world_size=2, rank=0)
    model = dist_nn.DistributedDataParallel(module=model, evice_ids=[0], output_device=0)
    for i in loader:
        print(i)
        break
    dist.destroy_process_group()


