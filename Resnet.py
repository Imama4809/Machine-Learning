import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


#load the data in via a function 

def load_data():
    #left empty 
    pass


class Block(nn.Module):
    """
    this is a class used to create the block  
    """
    def __init__(self,filters,kernal_size, stride, padding,NOLIB, HT = False):
        """
        initialize
        
        Args:   
        filters (int): the number of filters
        kernal_size (int): the expected kernal size for each layer
        stride (int): the expected stride for each layer
        padding (int): the expected padding for each later
        NOLIB (int): number of layers in block
        HT (boolean): "Half True", does the block end by dividing the feature space by 2?
        
        output:
        x (tensor): resulting vector after passing through layers 
        """
        self.filters = filters
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        self.NOLIB = NOLIB
        self.HT = HT
        self.array_of_layers = []
        for _ in range(NOLIB-1):
            self.array_of_layers.append(create_instance())
        if not self.HT:
            self.array_of_layers.append(create_instance(pooling = True))
        else:
            self.array_of_layers.append(create_intance())
            
        
    def create_instance(self,pooling = False):
        """
        This is a helper function to make the layers, this helps making a variable number of layers 
        
        Args:
        pooling (boolean): this will be to decide whether the layer of the block is a pooling layer or not 
        
        output:
        layer (nn.Sequential): this is a layer
        """
        if not pooling:
            layer = nn.Sequential(
                nn.Conv2d(self.filters,self.filters,self.kernal_size,self.stride,self.padding,self.x),
                nn.BatchNorm2d(self.filters),
                nn.ReLU()
                )
        else:
            layer = nn.Sequential(
                nn.Conv2d(self.filters,self.filters,self.kernal_size,self.stride,self.padding,self.x),
                nn.BatchNorm2d(self.filters),
                nn.ReLU(),
                nn.MaxPool2d(kernal_size=3,stride=2)
                )
        return layer
    
    def pass_through_block(self,x):
        """
        passing x through the layers
        
        Args:
        x (tensor): the input tensor
        
        Output:
        x (tensor): the input tensor after passing through the layers 
        """
        original = x
        for layer in self.array_of_layers:
            x = layer(x)
        x = x+original
        return x 
    


class Resnet(nn.Module):
    """
    Resnet model
    """
    def __init__(self):
        """
        initalize
        """
        #left empty to add in variables later
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.linear = nn.Linear()
        
    def forward(self,x):
        """
        forward
        
        Args:
        x (tensor): the input tensor
        
        Output:
        x (tensor): the output tensor
        """
        x = self.block1.pass_through_block(x)
        x = self.block2.pass_through_block(x)
        x = self.block3.pass_through_block(x)
        x = self.linear
        return x
    


def train_model():
    #left empty 
    pass
def test_model():
    #left empty
    pass 



        