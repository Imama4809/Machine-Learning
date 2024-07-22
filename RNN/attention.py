import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import math
import copy


def attention(key, query, value):
    d_k = query.size(-1)
    kq_pair = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    kq_pair = kq_pair.softmax(dim=-1)
    return torch.matmul(kq_pair,value), kq_pair

class Mulithead_Attention(nn.Module):
    def __init__(self,d_m, d_k, d_v, heads ):
        super(Mulithead_Attention,self).__init__()
        self.d_m = d_m
        self.d_k = d_k
        self.d_q = d_k
        self.d_v = d_v
        self.heads = heads 
        self.key = nn.Parameter(torch.rand(d_m,d_k))
        self.query = nn.Parameter(torch.rand(d_m,d_k))
        self.value = nn.Parameter(torch.rand(d_m,d_v))
        self.out = nn.Parameter(torch.rand(heads*d_v,d_m))
        self.keys = []
        self.querys = []
        self.values = []
        for val in range(head):
            self.keys.append(copy.deepcopy(self.key))
            self.queries.append(copy.deepcopy(self.query))
            self.values.append(copy.deepcopy(self.value))
        #not sure if while training this would result in the same updates to each tensor
    def forward(self,x):
        x = x.size(-1,d_m)
        for inc in range(heads):
            key = torch.matmul(x,self.keys[inc])
            query = torch.matmul(x,self.queries[inc])
            value = torch.matmul(x,self.values[inc])
            result = attention(key,query,value)
            if (inc == 0):
                total = result
            else:
                total = torch.cat((torch,result), dim=1)
        return torch.matmul(total, self.out)
        # for _ in range(heads):
            
