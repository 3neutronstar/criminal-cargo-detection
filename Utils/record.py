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
        self.test_csv=pd.read_csv(data_path,'train.csv')
        from DataProcessing.preprocessing import Preprocessing
        Preprocessing(self.test_csv,configs)
        self.npy_dict=Preprocessing.run()
        self.data_loader=DataLoader(self.npy_dict['table_data'])
        self.configs=load_params(configs,current_path,configs['file_name'])

        self.input_space=self.dataloader.dataset[0][0].size()[0]
        if 'crime' in configs['mode']:
            self.output_space=2
        elif 'priority' in configs['mode']:
            self.output_space=2
        else: #mixed
            self.output_space=2
        self.model=MODEL[configs['mode'].split('_')[1]](self.data_loader.dataset,self.output_space,configs)
        self.metric={}
    
    def run(self):
        self._load_model()
        self.model.to(self.configs['device'])

        if 'crime' in self.configs['mode'] or 'priority' in self.configs['mode']:
            metric=self.run_ind()
        else: #mixed
            metric=self.run_mix()
        
        self._record(metric)
        
    def run_mix(self):
        self.model.eval()        
        
        with torch.no_grad():
            for batch_idx,(data,crime_targets,priority_targets) in enumerate(self.data_loader):
                data,crime_targets,priority_targets=data.to(self.configs['device']),crime_targets.to(self.configs['device']),priority_targets.to(self.configs['device'])

                crime_outputs,priority_outputs=self.model(data)
                crime_predictions=torch.max(crime_outputs,dim=1)[1].clone()
                priority_predictions=torch.max(priority_outputs,dim=1)[1].clone()

                priority_predictions[crime_predictions==0]=-1
                metric=self._save_score(crime_predictions,priority_predictions+1)
        return copy.deepcopy(metric)

    def run_ind(self):
        self.model.eval()

        with torch.no_grad():
            for batch_idx,(data,targets) in enumerate(self.test_dataloader):
                data,targets=data.to(self.configs['device']),targets.to(self.configs['device'])

                outputs=self.model(data)
                predictions=torch.max(outputs,dim=1)[1].clone() # cross-entropy
                # predictions=torch.round(outputs).view(-1)#linear regression

                if self.metric == dict():
                    self.metric['predictions']=predictions
                    self.metric['targets']=targets
                else:
                    self.metric['predictions']=torch.cat((self.metric['predictions'],predictions),dim=0)
                    self.metric['targets']=torch.cat((self.metric['targets'],targets),dim=0)
        return copy.deepcopy(self.metric)

    def _record(self,metric):
        self.test_csv['우범여부']=metric['crime']['predictions']
        self.test_csv['핵심적발']=metric['priority']['predictions']
        self.test_csv.to_csv(os.path.join(self.save_path,self.configs['file_name']+'_test.csv'))
        return

    def _load_model(self):
        load_model_state=os.path.join(self.save_path,self.configs['file_name'],self.configs['file_name']+'.pt')
        dict_model=torch.load(load_model_state)
        print('Model\'s Score: ')
        for key in dict_model.keys():
            if key=='model_state_dict':
                continue
            else:
                print(dict_model[key])
        self.model.load_state_dict(dict_model['model_state_dict'])
        

    def save_models(self,epoch, score_dict):
        dict_model={
            'epoch':epoch,
            'model_state_dict':self.model.state_dict(),
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
            