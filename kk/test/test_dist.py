import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed as dist_data
import torch.nn.parallel.distributed as dist_nn
import torch.utils.data as data
import tqdm


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.net = nn.Linear(5, 1)

    def forward(self, x):
        return self.net(x)


class User_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(User_Dataset, self).__init__()
        self.x = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        self.y = torch.ones(10, dtype=torch.float32)
        self.data = list(zip(self.x, self.y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(data)


def test():
    dataset = User_Dataset()
    sampler = dist_data.DistributedSampler(dataset, num_replicas=2, rank=0)
    loader = data.DataLoader(dataset, batch_size=2, sampler=sampler)
    model = TestModel()
    dist.init_process_group(backend="nccl", init_method="env://", world_size=2, rank=0)
    model = dist_nn.DistributedDataParallel(module=model, evice_ids=[0], output_device=0)
    for i in tqdm(loader):
        print(i)
        break
    dist.destroy_process_group()


if __name__ == "__main__":
    test()

