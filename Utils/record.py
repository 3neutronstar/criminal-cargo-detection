import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import os

from torch.utils.data.dataset import Subset, TensorDataset
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
        test_dataset=TensorDataset(torch.from_numpy(self.npy_dict['test_crime_data']).float(),torch.from_numpy(self.npy_dict['test_priority_data']).float())
        self.data_loader=DataLoader(test_dataset)
        self.configs=load_params(configs,current_path,configs['file_name'])
        self.configs['file_name']=configs['file_name']
        if 'crime' in configs['mode']:
            self.input_space=self.data_loader.dataset[0][0].size()[0]
            self.output_space=2
        elif 'priority' in configs['mode']:
            self.input_space=self.data_loader.dataset[0][0].size()[0]
            self.output_space=2
        else: #mixed
            self.input_space=[self.data_loader.dataset[0][0].size()[0], self.data_loader.dataset[0][1].size()[0]]
            self.output_space=2
        self.model=MODEL[self.configs['mode'].split('_')[1]](self.input_space,self.output_space,configs)
        self.metric={}
        self.scoring_metric={}
    
    def _eval_before_save(self):
        mixed_dataset=TensorDataset(torch.from_numpy(self.npy_dict['train_crime_data']).float(),torch.from_numpy(self.npy_dict['train_priority_data']).float(),torch.from_numpy(self.npy_dict['crime_targets']).long(),torch.from_numpy(self.npy_dict['priority_targets']).long())
        # train_dataset=Subset(mixed_dataset,self.npy_dict['train_indices'])
        valid_dataset=Subset(mixed_dataset,self.npy_dict['valid_indices'])
        valid_dataloader=DataLoader(valid_dataset,batch_size=self.configs['batch_size'])
        self.model.eval()
        eval_loss=0.0
        score_dict={
            'loss':0.0,
            'total':0.0,
            'crime':{
            'accuracy':0.0,
            'total':0.0,
            'precision':0.0,
            'recall':0.0,
            'f1score':0.0,
            'loss':0.0,
            'custom_loss':0.0,
        },
            'priority':{
            'accuracy':0.0,
            'total':0.0,
            'precision':0.0,
            'recall':0.0,
            'f1score':0.0,
            'loss':0.0,
            'custom_loss':0.0,
        },
            'advantage':0.0
        }
        print("Start Eval")
        self.criterion=self.model.criterion
        with torch.no_grad():
            for batch_idx,(crime_data,priority_data,crime_targets,priority_targets) in enumerate(valid_dataloader):
                crime_data,priority_data=crime_data.to(self.configs['device']),priority_data.to(self.configs['device'])
                crime_targets,priority_targets=crime_targets.to(self.configs['device']),priority_targets.to(self.configs['device'])
                
                crime_outputs,priority_outputs=self.model(crime_data,priority_data)
                crime_loss,priority_loss=self.criterion(crime_outputs,priority_outputs,crime_targets,priority_targets)
                crime_predictions=torch.max(crime_outputs,dim=1)[1].clone()
                priority_predictions=torch.max(priority_outputs,dim=1)[1].clone()

                #loss
                loss=crime_loss+priority_loss
                score_dict['crime']['loss']+=crime_loss.item()
                score_dict['priority']['loss']+=priority_loss.item()
                priority_predictions[crime_predictions==0]=-1
                self._save_scoring(crime_predictions,crime_targets,priority_predictions+1,priority_targets)
                eval_loss +=loss.item()
        # crime
        score_dict['crime']=self._get_score(self.scoring_metric['crime']['predictions'],self.scoring_metric['crime']['targets'],score_dict['crime'])
        # priority
        score_dict['priority']=self._get_score(self.scoring_metric['priority']['predictions'],self.scoring_metric['priority']['targets'],score_dict['priority'])
                
        score_dict['loss']=eval_loss/(batch_idx+1)
        score_dict['crime']['loss']=score_dict['crime']['loss']/(batch_idx+1)
        score_dict['priority']['loss']=score_dict['priority']['loss']/(batch_idx+1)
        score_dict['advantage'] = (score_dict['crime']['f1score'] + score_dict['priority']['f1score'])*0.5
        print('[Crime Acc] {0:.4f}\t [Priority Acc] {1:.4f}\t'.format(score_dict['crime']['accuracy'], score_dict['priority']['accuracy']))
        print('[Crime F1] {0:.4f}\t [Priority F1] {1:.4f}\t [Advantage] {2:.4f}\t'.format(score_dict['crime']['f1score'], score_dict['priority']['f1score'], score_dict['advantage']))
        print("End Eval")

    def run(self):
        self._load_model()
        print("Model Load Complete")
        self.model.to(self.configs['device'])
        # eval하는거 성능측정 확인
        #self._eval_before_save()

        if 'mixed' not in self.configs['mode']:
            metric=self.run_ind()
        else: #mixed
            metric=self.run_mix()
        print("Prediction Complete")
        
        self._record(metric)
        print("Record Complete")
        
    def run_mix(self):
        self.model.eval()        
        metric=dict()
        with torch.no_grad():
            for batch_idx,(crime_data,priority_data) in enumerate(self.data_loader):
                crime_data=crime_data.to(self.configs['device'])
                priority_data=priority_data.to(self.configs['device'])

                crime_outputs,priority_outputs=self.model(crime_data,priority_data)
                crime_predictions=torch.max(crime_outputs,dim=1)[1].clone()
                priority_predictions=torch.max(priority_outputs,dim=1)[1].clone()

                priority_predictions[crime_predictions==0]=-1
                metric=self._save_score(crime_predictions,priority_predictions+1,metric)
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
        print('Checking length of test.csv :', len(self.test_csv.index))
        self.test_csv.to_csv(os.path.join(self.save_path,self.configs['file_name']+'_test.csv'), index = False)
        return
        
    def _load_model(self):
        if 'mixed' not in self.configs['mode']:
            dict_model=torch.load(os.path.join(self.save_path,self.configs['file_name'],'best_{}_model.pt').format(self.configs['mode'].split('_')[1]))
            self.model.load_model(dict_model)
        else: 
            crime_dict=copy.deepcopy(torch.load(os.path.join(self.save_path,self.configs['file_name'],'best_crime_model.pt')))
            priority_dict=copy.deepcopy(torch.load(os.path.join(self.save_path,self.configs['file_name'],'best_priority_model.pt')))
            # print("========== Performances ==========")
            # print("crime F1: {:.3f} crime Acc: {:.3f}".format(crime_dict['f1score'],crime_dict['accuracy']))
            # print("priority F1: {:.3f} priority Acc: {:.3f}".format(priority_dict['f1score'],priority_dict['accuracy']))
            # print("Advantage: {:.3f}".format((crime_dict['f1score']+priority_dict['f1score'])*0.5))
            # print("==================================")
            
            # Modified
            dict_model={**crime_dict,**priority_dict}
            self.model.load_model(dict_model)

            # Origin
            # self.model.load_model(crime_dict,priority_dict)
        

    def save_models(self,epoch, score_dict):
        dict_model={
            'epoch':epoch,
            '{}_model_state_dict'.format(self.configs['mode'].split('_')[1]):self.model.state_dict(),
        }.update(score_dict)
        torch.save(dict_model,
        os.path.join(self.save_path,self.time_data,'best_model.pt'))
    
    def _save_score(self,crime_predictions,priority_predictions,metric):
        if metric==dict():
            metric['crime']=dict()
            metric['priority']=dict()
            metric['crime']['predictions']= crime_predictions
            metric['priority']['predictions']= priority_predictions
        else:
            metric['crime']['predictions']=torch.cat((metric['crime']['predictions'],crime_predictions),dim=0)
            metric['priority']['predictions']=torch.cat((metric['priority']['predictions'],priority_predictions),dim=0)

        return metric
    
    def _save_scoring(self,crime_predictions,crime_targets,priority_predictions,priority_targets):
        if self.scoring_metric==dict():
            self.scoring_metric['crime']=dict()
            self.scoring_metric['priority']=dict()
            self.scoring_metric['crime']['predictions']= crime_predictions
            self.scoring_metric['crime']['targets']= crime_targets
            self.scoring_metric['priority']['predictions']= priority_predictions
            self.scoring_metric['priority']['targets']= priority_targets
        else:
            self.scoring_metric['crime']['predictions']=torch.cat((self.scoring_metric['crime']['predictions'],crime_predictions),dim=0)
            self.scoring_metric['crime']['targets']=torch.cat((self.scoring_metric['crime']['targets'],crime_targets),dim=0)
            self.scoring_metric['priority']['predictions']=torch.cat((self.scoring_metric['priority']['predictions'],priority_predictions),dim=0)
            self.scoring_metric['priority']['targets']=torch.cat((self.scoring_metric['priority']['targets'],priority_targets),dim=0)

        return self.metric
    
    def _save_scoring(self,crime_predictions,crime_targets,priority_predictions,priority_targets):
        if self.scoring_metric==dict():
            self.scoring_metric['crime']=dict()
            self.scoring_metric['priority']=dict()
            self.scoring_metric['crime']['predictions']= crime_predictions
            self.scoring_metric['crime']['targets']= crime_targets
            self.scoring_metric['priority']['predictions']= priority_predictions
            self.scoring_metric['priority']['targets']= priority_targets
        else:
            self.scoring_metric['crime']['predictions']=torch.cat((self.scoring_metric['crime']['predictions'],crime_predictions),dim=0)
            self.scoring_metric['crime']['targets']=torch.cat((self.scoring_metric['crime']['targets'],crime_targets),dim=0)
            self.scoring_metric['priority']['predictions']=torch.cat((self.scoring_metric['priority']['predictions'],priority_predictions),dim=0)
            self.scoring_metric['priority']['targets']=torch.cat((self.scoring_metric['priority']['targets'],priority_targets),dim=0)

    def _get_score(self,predictions,targets,score_dict):
        from Utils.calc_score import calc_score
        score_dict=calc_score(predictions,targets,score_dict)
        return score_dict