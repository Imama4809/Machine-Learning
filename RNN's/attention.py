import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import math


def attention(query, key, value):
    d_k = query.size(-1)
    kq_pair = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    kq_pair = kq_pair.softmax(dim=-1)
    return torch.matmul(kq_pair,value), kq_pair



