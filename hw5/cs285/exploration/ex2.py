from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

# def init_method_1(model):
#     model.weight.data.uniform_()
#     model.bias.data.uniform_()

# def init_method_2(model):
#     model.weight.data.normal_()
#     model.bias.data.normal_()


class EX2Model(nn.Module, BaseExplorationModel):
    