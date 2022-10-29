import torch

batch_size = 128
clipped = torch.Tensor([2])
scale = torch.exp(clipped)
print(scale)