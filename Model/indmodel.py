import torch
import torch.nn as nn
from Utils.custom_loss import loss_kd_regularization,f_beta_score_loss


class CustomLossFunction():
    def __init__(self,criterion,configs):
        self.criterion=criterion
        self.configs=configs
    
    def __call__(self,outputs,ground_truth):
        if self.configs['custom_loss']=='kd_loss':
            loss=loss_kd_regularization(outputs,ground_truth)
        elif self.configs['custom_loss']=='fbeta_loss':
            f1_loss=f_beta_score_loss(outputs,ground_truth)
            loss=self.criterion(outputs,ground_truth)
            loss+=f1_loss
        else:
            loss=self.criterion(outputs,ground_truth)
        return loss


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
        self.criterion=CustomLossFunction(nn.CrossEntropyLoss(),configs)
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
            'model_state_dict':self.model.state_dict(),
        }.update(score_dict)
        return dict_model

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
            nn.Linear(100,output_space),
        )
        self.criterion=CustomLossFunction(nn.CrossEntropyLoss(),configs)
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
            'model_state_dict':self.model.state_dict(),
        }.update(score_dict)
        return dict_model

    def load_model(self,dict_model):
        self.load_state_dict(dict_model['crime_model_state_dict'])