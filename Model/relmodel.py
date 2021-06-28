import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim.lr_scheduler import StepLR
from Model.indmodel import CrimeModel,PriorityModel

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
    
    def step(self):
        self.crime_optim.step()
        self.priority_optim.step()

    def zero_grad(self):
        self.priority_optim.zero_grad()
        self.crime_optim.zero_grad()

class MixedLossFunction():
    def __init__(self,crime_criterion,priority_criterion):
        self.crime_criterion=crime_criterion
        self.priority_criterion=priority_criterion
    
    def __call__(self,crime_y_pred,priority_y_pred,crime_y_truth,priority_y_truth):
        crime_loss=self.crime_criterion(crime_y_pred,crime_y_truth)
        priority_loss=self.priority_criterion(priority_y_pred,priority_y_truth)
        return crime_loss,priority_loss

class MixedModel(nn.Module):
    def __init__(self,input_space,output_space,configs):
        super(MixedModel,self).__init__()
        self.crime_model=CrimeModel(input_space,output_space,configs)
        self.priority_model=PriorityModel(input_space+output_space,output_space,configs)
        self.criterion=MixedLossFunction(self.crime_model.criterion,self.priority_model.criterion)
        self.optimizer=MixedOptimizer(self.crime_model.optimizer,self.priority_model.optimizer)
        self.scheduler=MixedScheduler(self.crime_model.scheduler,self.priority_model.scheduler)

    def forward(self,x):
        crime_output=self.crime_model(x)
        softened_crime_output=f.softmax(crime_output,dim=1)
        priority_input=torch.cat((softened_crime_output,x),dim=1)
        priority_output=self.priority_model(priority_input)
        return crime_output,priority_output
