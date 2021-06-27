import torch
import torch.nn as nn

class CrimeModel(nn.Module):
    def __init__(self,input_space,output_space,configs):
        super(CrimeModel,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(input_space,5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(5000,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,output_space)
        )
        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=configs['lr'],weight_decay=configs['weight_decay'])
        self.scheduler=torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,step_size=configs['lr_decay'], gamma=configs['lr_decay_rate'])

        for m in self.modules():
            if isinstance(m,(nn.Linear)):
                nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        y=self.model(x)
        return y

class PriorityModel(nn.Module):
    def __init__(self,input_space,output_space,configs):
        super(PriorityModel,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(input_space,5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(5000,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,output_space)
        )
        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=configs['lr'],weight_decay=configs['weight_decay'])
        self.scheduler=torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,step_size=configs['lr_decay'], gamma=configs['lr_decay_rate'])

        for m in self.modules():
            if isinstance(m,(nn.Linear)):
                nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        y=self.model(x)
        return y