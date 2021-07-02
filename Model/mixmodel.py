import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.modules import loss
from torch.optim.lr_scheduler import StepLR
from Model.indmodel import CrimeModel,PriorityModel
from Utils.custom_loss import *
class MixedScheduler():
    def __init__(self,crime_scheduler,priority_scheduler):
        self.crime_scheduler=crime_scheduler
        self.priority_scheduler=priority_scheduler

    def step(self):
        self.crime_scheduler.step()
        self.priority_scheduler.step()

class MixedOptimizer():
    def __init__(self,crime_optim,priority_optim):
        self.crime_optim=crime_optim
        self.priority_optim=priority_optim
        self.param_groups=self.crime_optim.param_groups
    
    def step(self):
        self.crime_optim.step()
        self.priority_optim.step()

    def zero_grad(self):
        self.priority_optim.zero_grad()
        self.crime_optim.zero_grad()

class MixedLossFunction():
    def __init__(self,crime_criterion,priority_criterion,configs):
        self.crime_criterion=crime_criterion
        self.priority_criterion=priority_criterion
        self.configs=configs
        if self.configs['custom_loss']=='kd_loss':
            self.custom_criterion=KDRegLoss(configs)
        elif self.configs['custom_loss']=='fbeta_loss':
            self.custom_criterion=FBetaLoss(configs)
        else: 
            self.custom_criterion=None
    
    def __call__(self,crime_y_pred,priority_y_pred,crime_y_truth,priority_y_truth):
        crime_custom_loss=0.0
        priority_custom_loss=0.0

        crime_loss=self.crime_criterion(crime_y_pred,crime_y_truth)
        if self.custom_criterion is not None:
            crime_custom_loss=self.custom_criterion(crime_y_pred,crime_y_truth)
        crime_loss+=crime_custom_loss
        
        # search only
        idx=torch.logical_or(priority_y_truth==1,priority_y_truth==2)
        priority_y_truth=priority_y_truth[idx]-1
        priority_y_pred=priority_y_pred[torch.stack((idx,idx),dim=1)].view(-1,2)
        priority_loss=torch.nan_to_num(self.priority_criterion(priority_y_pred,priority_y_truth))
        if self.custom_criterion is not None:
            priority_custom_loss=torch.nan_to_num(self.custom_criterion(priority_y_pred,priority_y_truth))
        priority_loss+=priority_custom_loss

        return crime_loss,priority_loss

class MixedModel(nn.Module):
    def __init__(self,input_space,output_space,configs):
        super(MixedModel,self).__init__()
        self.crime_model=CrimeModel(input_space,output_space,configs)
        self.priority_model=PriorityModel(input_space+output_space,output_space,configs)#
        self.criterion=MixedLossFunction(self.crime_model.criterion,self.priority_model.criterion,configs)
        self.optimizer=MixedOptimizer(self.crime_model.optimizer,self.priority_model.optimizer)
        self.scheduler=MixedScheduler(self.crime_model.scheduler,self.priority_model.scheduler)

    def forward(self,x):
        crime_output=self.crime_model(x)
        softened_crime_output=f.softmax(crime_output,dim=1).detach().clone()
        priority_input=torch.cat((softened_crime_output,x),dim=1)
        # priority_input=x
        priority_output=self.priority_model(priority_input)
        return crime_output,priority_output

    def save_model(self,epoch,score_dict):
        dict_model={
            'epoch':epoch,
            'crime_model_state_dict':self.crime_model.state_dict(),
            'priority_model_state_dict':self.priority_model.state_dict(),
        }
        dict_model.update(score_dict)

        return dict_model

    def load_model(self,dict_model):
        self.crime_model.load_state_dict(dict_model['crime_model_state_dict'])
        self.priority_model.load_state_dict(dict_model['priority_model_state_dict'])