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

class LSTM_CELL(nn.Module):
    def __init__(self, C_s, h_s, x_t): #in the alphabet, s comes right before t, thus t-1 = s
        super(LSTM,self).__init__()
        self.C_s = C_s
        self.h_s = h_s
        self.x_t = x_t
        self.xhconcat = torch.concat((self.x_t,self.h_s),dim=0)
        
        #forget gate weight
        self.W_f = nn.Parameter(torch.random(self.x_t.size(0),self.x_t.size(0)+self.h_s.size(0)))
        self.B_f = nn.Parameter(torch.random(self.x_t.size(0)))
        
        #input gate weight
        self.W_i = nn.Parameter(torch.random(self.x_t.size(0),self.x_t.size(0)+self.h_s.size(0)))
        self.B_i = nn.Parameter(torch.random(self.x_t.size(0)))
        
        self.W_C = nn.Parameter(torch.random(self.x_t.size(0),self.x_t.size(0)+self.h_s.size(0)))
        self.B_C = nn.Parameter(torch.random(self.x_t.size(0)))
        
        self.W_o = nn.Parameter(torch.random(self.x_t.size(0),self.x_t.size(0)+self.h_s.size(0)))
        self.B_o = nn.Parameter(torch.random(self.x_t.size(0)))
        
    def forward(self,x = self.x_t):
        #forget gate
        f_t = torch.sigmoid(torch.matmul(self.W_f,self.xhconcat) + self.B_f)
        
        #input gate
        i_t = torch.sigmoid(torch.matmul(self.W_i,self.xhconcat) + self.B_i)
        C_t = torch.tanh((torch.matmul(self.W_C,self.xhconcat) + self.B_C))
        #i assume its element wise multiplication 
        C_t = f_t * self.C_s + i_t*C_t
        
        #output gate
        O_t = torch.sigmoid(torch.matmul(self.W_o,self.xhconcat) + self.B_o)
        h_t = O_t * torch.tanh(C_t)
        
        return h_t,C_t
        

