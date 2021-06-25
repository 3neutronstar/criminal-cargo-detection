import torch
import torch.nn as nn
class MixedModel(nn.Module):
    def __init__(self,input_space,output_space,configs):
        super(MixedModel,self).__init__()