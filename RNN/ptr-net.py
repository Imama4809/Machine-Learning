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



class encoder(nn.Module):
    def __init__(self, input_size, hidden_state, cell_state, num_layers):
        self.H_t = hidden_state
        self.C_t = cell_state
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers)
    def forward(self,inp):
        output, (new_hidden_state, new_cell_state) =  self.encoder(inp,(self.H_t,self.C_t))
        return output, new_hidden_state, new_cell_state
        
        
class decoder(nn.Module):
    def __init__(self, input_size, hidden_state, cell_state, num_layers):
        self.H_t = hidden_state
        self.C_t = cell_state
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers)
    def forward(self,inp):
        output, (new_hidden_state, new_cell_state) =  self.encoder(inp,(self.H_t,self.C_t))
        return output, new_hidden_state, new_cell_state
        

class CBIA(nn.Module): #Content Based Input Attention
    def __init__(self,input_size,hidden_state,cell_state, num_layers, inp_length, trg_length):
        self.input_size = input_size
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        self.hidden_size = self.hidden_state.size()
        self.num_layers = num_layers
        self.e = encoder(input_size,self.hidden_state,self.cell_state,self.num_layers)
        self.h_states = [self.hidden_state]
        for inp in inp_length:
            output, (self.hidden_state,self.cell_state) = self.e(inp,(self.hidden_state,self.cell_state))
            self.h_states.append(self.hidden_state)
    def forward(self,inp):
        self.d = decoder(inp.size(),self.hidden_state,self.cell_state,self.num_layers)
        
        
        
        return unembed(d_outputs)