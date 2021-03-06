import torch
import torch.nn as nn
from Utils.custom_loss import CrossEntropyLoss_weighted, FocalLoss
class CrimeModel(nn.Module):
    def __init__(self,input_space,output_space,configs):
        super(CrimeModel,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(input_space,5000),
            nn.BatchNorm1d(5000),#5000),
            nn.ReLU(),
            nn.Linear(5000,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000,300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(300,output_space)
        )
        self.weight1 = torch.tensor([0.2988, 1.0])
        self.criterion=nn.CrossEntropyLoss(weight=self.weight1)
        # self.criterion=nn.CrossEntropyLoss()
        # self.criterion=CrossEntropyLoss_weighted()
        # self.criterion=FocalLoss(gamma=0)
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


    def save_model(self,epoch,score_dict):
        dict_model={
            'epoch':epoch,
            'crime_model_state_dict':self.model.state_dict(),
        }
        dict_model.update(score_dict)
        return dict_model

    def load_model(self,dict_model):
        self.model.load_state_dict(dict_model['crime_model_state_dict'])

class PriorityModel(nn.Module):
    def __init__(self,input_space,output_space,configs):
        super(PriorityModel,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(input_space,5000),
            nn.BatchNorm1d(5000),#5000),
            nn.ReLU(),
            nn.Linear(5000,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000,300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(300,output_space)
        )
        self.weight2 = torch.tensor([1.0, 0.9431])
        self.criterion=nn.CrossEntropyLoss(weight=self.weight2)
        # self.criterion=nn.CrossEntropyLoss()
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

    def save_model(self,epoch,score_dict):
        dict_model={
            'epoch':epoch,
            'priority_model_state_dict':self.model.state_dict(),
        }
        dict_model.update(score_dict)
        return dict_model

    def load_model(self,dict_model):
        self.load_state_dict(dict_model['priority_model_state_dict'])