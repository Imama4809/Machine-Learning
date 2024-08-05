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


def embed(values):
    #this is a function that will be changed depending on the specific usage 
    return values
def unembed(values):
    #this is a function that will be changed depending on the specific usage 
    return values


class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        self.H_0 = 0
        self.C_0 = 0
        encoder = nn.LSTM(input_size, hidden_size, num_layers)
    def forward(self,inp):
        
        #for seq2seq model
        inputs = embed(inputs)
        output, (hidden, cell) = encoder(inputs,(self.H_0,self.C_0))
        return output, hidden, cell
        
        
class decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, encoder_H_t, encoder_C_0, trg_length):
        self.trg_length = trg_length
        self.H_0 = encoder_H_t
        self.C_0 = encoder_C_0
        decoder = nn.LSTM(input_size,hidden_size,num_layers)
    def forward(self,inp):
        outputs = []
        hidden_state = self.H_0
        for _ in range(self.trg_length):
            output, (hidden,cell) = encoder(inp,(hidden_state,self.C_0))
            hidden_state = hidden
            outputs.append(output)
        return outputs, hidden, cell
        
class seq2seq(nn.Module):
    def __init__(self,input_size, hidden_size,num_layers):
        self.e = encoder(input_size,hidden_size,num_layers)
    def forword(self,inp):
        e_output, e_hidden_state, e_cell_state = self.e(inp)
        self.d = decoder(e_output,e_hidden_state,e_cell_state)