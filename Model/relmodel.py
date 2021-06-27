import torch
import torch.nn as nn
from Model.indmodel import CrimeModel,PriorityModel
class MixedModel(nn.Module):
    def __init__(self,input_space,output_space,configs):
        super(MixedModel,self).__init__()
        self.crime_model=CrimeModel(input_space,output_space,configs)
        self.priority_model=PriorityModel(input_space,output_space,configs)
    def forward(self,x):
        
