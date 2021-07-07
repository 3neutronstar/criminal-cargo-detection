import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from Utils.params import load_params
from Model.basemodel import MODEL
import copy

class RecordData:
    def __init__(self,data_path,save_path,current_path,configs):
        self.configs=configs
        self.save_path=save_path
        self.test_csv=copy.deepcopy(pd.read_csv(os.path.join(data_path,'test.csv')))
        from DataProcessing.load_data import load_dataset
        self.npy_dict=load_dataset(data_path,configs)
        self.data_loader=DataLoader(torch.from_numpy(self.npy_dict['test_data']).float())
        self.configs=load_params(configs,current_path,configs['file_name'])
        self.configs['file_name']=configs['file_name']
        self.input_space=len(self.data_loader.dataset[0])
        if 'crime' in configs['mode']:
            self.output_space=2
        elif 'priority' in configs['mode']:
            self.output_space=2
        else: #mixed
            self.output_space=2
        self.model=MODEL[self.configs['mode'].split('_')[1]](self.input_space,self.output_space,configs)
        self.metric={}
    
    def run(self):
        self._load_model()
        print("Model Load Complete")
        self.model.to(self.configs['device'])

        if 'crime' in self.configs['mode'] or 'priority' in self.configs['mode']:
            metric=self.run_ind()
        else: #mixed
            metric=self.run_mix()
        print("Prediction Complete")
        
        self._record(metric)
        print("Record Complete")
        
    def run_mix(self):
        self.model.eval()        
        
        with torch.no_grad():
            for batch_idx,data in enumerate(self.data_loader):
                data=data.to(self.configs['device'])

                crime_outputs,priority_outputs=self.model(data)
                crime_predictions=torch.max(crime_outputs,dim=1)[1].clone()
                priority_predictions=torch.max(priority_outputs,dim=1)[1].clone()

                priority_predictions[crime_predictions==0]=-1
                metric=self._save_score(crime_predictions,priority_predictions+1)
        return copy.deepcopy(metric)

    def run_ind(self):
        self.model.eval()

        with torch.no_grad():
            for batch_idx,data in enumerate(self.data_loader):
                data=data.to(self.configs['device'])

                outputs=self.model(data)
                predictions=torch.max(outputs,dim=1)[1].clone() # cross-entropy
                # predictions=torch.round(outputs).view(-1)#linear regression

                if self.metric == dict():
                    self.metric['predictions']=predictions
                else:
                    self.metric['predictions']=torch.cat((self.metric['predictions'],predictions),dim=0)
        return copy.deepcopy(self.metric)

    def _record(self,metric):
        for key in self.test_csv.columns:
            if key not in ['우범여부','핵심적발','신고번호']:
                self.test_csv.drop(key,axis=1,inplace=True)
        if self.configs['mode'].split('_')[1]=='mixed':
            self.test_csv['우범여부']=metric['crime']['predictions'].cpu()
            self.test_csv['핵심적발']=metric['priority']['predictions'].cpu()
        elif self.configs['mode'].split('_')[1]=='crime':
            self.test_csv['우범여부']=metric['predictions'].cpu()
        elif self.configs['mode'].split('_')[1]=='priority':
            self.test_csv['핵심적발']=metric['predictions'].cpu()
        self.test_csv.to_csv(os.path.join(self.save_path,self.configs['file_name']+'_test.csv'))
        return
        
    def _load_model(self):
        dict_model=torch.load(os.path.join(self.save_path,self.configs['file_name'],'best_model.pt'))
        self.model.load_model(dict_model)

    def save_models(self,epoch, score_dict):
        dict_model={
            'epoch':epoch,
            '{}_model_state_dict'.format(self.configs['mode'].split('_')[1]):self.model.state_dict(),
        }.update(score_dict)
        torch.save(dict_model,
        os.path.join(self.save_path,self.time_data,'best_model.pt'))
    
    def _save_score(self,crime_predictions,priority_predictions):
        if self.metric==dict():
            self.metric['crime']=dict()
            self.metric['priority']=dict()
            self.metric['crime']['predictions']= crime_predictions
            self.metric['priority']['predictions']= priority_predictions
        else:
            self.metric['crime']['predictions']=torch.cat((self.metric['crime']['predictions'],crime_predictions),dim=0)
            self.metric['priority']['predictions']=torch.cat((self.metric['priority']['predictions'],priority_predictions),dim=0)

        return self.metric
            