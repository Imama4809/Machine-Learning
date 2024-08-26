# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from torch.optim.lr_scheduler import StepLR
# from torch.utils.data import DataLoader
# import math
# import copy



# class SSP(nn.Module):
#     def __init__(self):
#         super(SSP,self).__init__()
#     def forward(self,x):
#         return torch.log(0.5*torch.exp(x) + 0.5)
        


# class SchNet(nn.Module):
#     def __init__(self, position_values,features):
#         self.position_values = position_values

#         self.features = features
#         self.I1 = Interaction(self.position_values,self.features)
#         self.I2 = Interaction(self.position_values,self.features)
#         self.I3 = Interation(self.position_values,self.features)
#         self.l5 = nn.Linear(64,32)
#         self.l6 = SSP()
#         self.l7 = nn.Linear(32,1)
#     def forward(self,X):
#         self.unique_atoms = list(set(X))
#         self.embeddings = nn.Parameter(torch.rand(len(unique_atoms)))
#         indexes = []
#         for atom in X:
#             indexes.append(self.unique_atoms.index(atom))
#         indexes = torch.tensor(indexes)
#         X = self.embeddings[indexes]
#         X = self.I1(X)
#         X = self.I2(X)
#         X = self.I3(X)
#         X = self.l5(X)
#         X = self.l6(X)
#         X = self.l7(X) 
#         X = torch.sum(X, dim=0)
#         return X

# class Interaction(nn.Module):
#     def __init__(self, position_values ,features):
#         self.position_values = position_values
#         self.features = features
#         self.l1 = nn.Linear(64,64)
#         self.cfconv2 = cfconv(position_values ,features)
#         self.l3 = nn.Linear(64,64)
#         self.l4 = nn.SSP()
#         self.l5 = nn.Linear(64,64)
#     def forward(self,X):
#         og_X = copy.deepcopy(X)
#         X = self.l1(X)
#         X = self.cfconv2(X)
#         X = self.l3(X)
#         X = self.l4(X)
#         X = self.l5(X)
#         X = X + og_X
#         return X 
    


# # def radial_basis_func()

# class cfconv(nn.Module):
#     def __init__(self, position_values, features):
#         self.position_values = position_values
#         self.NOA = len(self.position_values)
#         self.features = features
#         self.centers = torch.nn.Parameter(torch.randn(300))
#         self.dense1 = nn.Linear(300,64)
#         self.ssp1 = SSP()
#         self.dense2 = nn.Linear(64,64)
#         self.ssp2 = SSP()
#     def forward(self,X):
#         d = []
#         for fa in range(len(position_values)):
#             for sa in range(fa+1,len(position_values)):
#                 dist = torch.sqrt(
#                     torch.square(position_values[fa][0]-position_values[sa][0])+
#                     torch.square(position_values[fa][1]-position_values[sa][1])+
#                     torch.square(position_values[fa][2]-position_values[sa][2])
#                     )
#                 d.append(dist)    
#         d = torch.tensor(d)
#         R = []
#         for dist in d:
#             for val in range(1,301):
#                 temparr = []
#                 temparr.append(torch.exp(-10*torch.square(dist-val/10)))
#                 R.append(temparr)
#         R = torch.tensor(R)
#         R = self.dense1(R)
#         R = self.ssp1(R)
#         R = self.dense2(R)
#         R = self.ssp2(R)
#         newX = torch.zeros_like(X)
#         i = 0
#         for fa in range(len(position_values)):
#             for sa in range(fa+1,len(position_values)):
#                 newX[fa] = newX[fa] + X[fa]*d[i]
#                 newX[sa] = newX[sa] + X[sa]*d[i]
#                 i=i+1
#         return newX
    
    
                
            
            
        
        
import torch
val = torch.rand(4,3)
print(val)
print(val + 2)
    
