from Model.mixmodel import MixedModel
import torch
from Utils.calc_score import calc_score
import time
import os
import logging
import copy
from Learner.baselearner import BaseLearner
from Model.basemodel import MODEL
from DataProcessing.load_data import load_dataloader
from torch.utils.tensorboard import SummaryWriter
from Utils.custom_loss import KDRegLoss,FBetaLoss

class TorchLearner(BaseLearner):
    def __init__(self, logger,time_data, data_path, save_path, device, configs):
        super(TorchLearner,self).__init__(logger,time_data, data_path, save_path, device, configs)
        self.train_dataloader,self.test_dataloader=load_dataloader(self.data_path,configs) # dataloader output(tensor) -> .numpy()
        self.input_space=self.train_dataloader.dataset[0][0].size()[0]
        if 'crime' in configs['mode']:
            self.output_space=2
        elif 'priority' in configs['mode']:
            self.output_space=2
        else: #mixed
            self.output_space=2
        self.model=MODEL[configs['mode'].split('_')[1]](self.input_space,self.output_space,configs)
        self.criterion=self.model.criterion
        self.optimizer=self.model.optimizer
        self.scheduler=self.model.scheduler
        self.logWriter=SummaryWriter(os.path.join(self.save_path,time_data))
        self.score_dict={
        'accuracy':0.0,
        'total':0.0,
        'precision':0.0,
        'recall':0.0,
        'f1score':0.0,
        'loss':0.0,
        'custom_loss':0.0,
        }
        self.best_f1score=0.0
        self.best_acc=0.0
        self.metric=dict()
        if self.configs['custom_loss']=='kd_loss':
            self.custom_criterion=KDRegLoss(configs)
        elif self.configs['custom_loss']=='fbeta_loss':
            self.custom_criterion=FBetaLoss(configs)
        else: 
            self.custom_criterion=None

    def run(self):
        self.model.to(self.configs['device'])
        self.logger.info(self.configs)

        for epoch in range(1,self.configs['epochs']+1):
            train_score_dict=copy.deepcopy(self.score_dict)
            eval_score_dict=copy.deepcopy(self.score_dict)
            #Init

            #Train
            self.metric=dict()# metric
            print('='*30)
            train_tik=time.time()
            train_score_dict=self._train(epoch,train_score_dict)
            train_tok=time.time()
            print('\n Learning Rate: {:.8f} Learning Time: {:.3f}s'.format(self.optimizer.param_groups[0]['lr'],train_tok-train_tik))
            self._epoch_end_logger(epoch,train_score_dict,'train')

            #Eval
            self.metric=dict()
            eval_score_dict=self._eval(epoch,eval_score_dict)
            self._epoch_end_logger(epoch,eval_score_dict,'eval')

            self.scheduler.step()
        
        if 'mixed' not in self.configs['mode']:
            self.logger = logging.getLogger('best')
            self.logger.info('[Mode {}] [Best Acc {:.2f}] [Best F1 {:.3f}]'.format(self.configs['mode'],self.best_acc,self.best_f1score))
        else:
            self.logger = logging.getLogger('best')
            self.logger.info('[Mode {}] [Best Crime Acc {:.2f}] [Best Crime F1 {:.3f}]'.format(self.configs['mode'],self.best_acc[0],self.best_f1score[0]))
            self.logger.info('[Mode {}] [Best Priority Acc {:.2f}] [Best Priority F1 {:.3f}]'.format(self.configs['mode'],self.best_acc[1],self.best_f1score[1]))
            advantage_score=0.5*self.best_f1score[0]+0.5*self.best_f1score[1]
            self.logger.info('[Advantage Score] {}'.format(advantage_score))

        print('==End==')

    def _train(self,epoch,score_dict):
        self.model.train()
        train_loss=0.0
        train_custom_loss=0.0
        custom_loss=None

        for batch_idx, (data,targets) in enumerate(self.train_dataloader):
            data,targets=data.to(self.configs['device']),targets.to(self.configs['device'])

            outputs=self.model(data)
            if self.custom_criterion is None:
                loss=self.criterion(outputs,targets)
            else:
                loss=self.criterion(outputs,targets)
                custom_loss=self.custom_criterion(outputs,targets)
                train_custom_loss+=custom_loss.item()
                loss+=custom_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss+=loss.item()
            # predictions=torch.round(outputs).view(-1)#linear regression
            predictions=torch.max(outputs,dim=1)[1].clone()
            if self.metric == dict():
                self.metric['predictions']=predictions
                self.metric['targets']=targets
            else:
                self.metric['predictions']=torch.cat((self.metric['predictions'],predictions),dim=0)
                self.metric['targets']=torch.cat((self.metric['targets'],targets),dim=0)
            if batch_idx%50==1:
                acc=(self.metric['predictions']==self.metric['targets']).sum()/self.metric['targets'].size(0)*100.0
                print('\r{}epoch {}/{}, [Acc] {:.2f} [Loss] {:.5f}'.format(epoch,self.metric['targets'].size(0),
                len(self.train_dataloader.dataset),acc,train_loss/(batch_idx+1)),end='')
            
        score_dict=self._get_score(self.metric['predictions'],self.metric['targets'],score_dict)
        score_dict['loss']=train_loss/(batch_idx+1)
        if custom_loss is not None:
            score_dict['custom_loss']=train_custom_loss/(batch_idx+1)
        return score_dict

  
    def _eval(self,epoch,score_dict):
        self.model.eval()
        eval_loss=0.0
        eval_custom_loss=0.0
        custom_loss=None

        with torch.no_grad():
            for batch_idx,(data,targets) in enumerate(self.test_dataloader):
                data,targets=data.to(self.configs['device']),targets.to(self.configs['device'])

                outputs=self.model(data)
                predictions=torch.max(outputs,dim=1)[1].clone() # cross-entropy
                # predictions=torch.round(outputs).view(-1)#linear regression
                if self.custom_criterion is None:
                    loss=self.criterion(outputs,targets)
                else:
                    loss=self.criterion(outputs,targets)
                    custom_loss=self.custom_criterion(outputs,targets)
                    eval_custom_loss+=custom_loss.item()
                    loss+=custom_loss

                eval_loss +=loss.item()
                if self.metric == dict():
                    self.metric['predictions']=predictions
                    self.metric['targets']=targets
                else:
                    self.metric['predictions']=torch.cat((self.metric['predictions'],predictions),dim=0)
                    self.metric['targets']=torch.cat((self.metric['targets'],targets),dim=0)
        score_dict=self._get_score(self.metric['predictions'],self.metric['targets'],score_dict)
        score_dict['loss']=eval_loss/(batch_idx+1)
        if custom_loss is not None:
            score_dict['custom_loss']=eval_custom_loss/(batch_idx+1)

        return score_dict

    def save_models(self,epoch, score_dict):
        dict_model= self.model.save_model(epoch,score_dict)
        print("save")
        torch.save(dict_model,
        os.path.join(self.save_path,self.time_data,'best_model.pt'))

    def load_model(self):
        dict_model=torch.load(self.save_path,self.configs['file_name'],'best_model.pt')
        self.model.load_model(dict_model)

    def _get_score(self,predictions,targets,score_dict):
        score_dict=calc_score(predictions,targets,score_dict)
        return score_dict
    
    def _epoch_end_logger(self,epoch,score_dict,mode='train'):
        
        if 'mixed' in self.configs['mode']:
            for model_type in ['crime','priority']:
                this_score_dict=score_dict[model_type]
                self._write_logger(epoch,model_type,this_score_dict,mode)
            if mode=='eval':
                save=False
                for i,model_type in enumerate(['crime','priority']):
                    if self.best_f1score[i]<score_dict[model_type]['f1score']:
                        save=True
                        self.best_f1score[i]=score_dict[model_type]['f1score']
                        self.best_acc[i]=score_dict[model_type]['accuracy']
                if save==True:
                    self.save_models(epoch,score_dict)
                    

        else:
            if mode=='eval':
                if self.best_f1score<score_dict['f1score']:
                    self.best_f1score=score_dict['f1score']
                    self.best_acc=score_dict['accuracy']
                    self.save_models(epoch,score_dict)
            self._write_logger(epoch,self.configs['mode'].split('_')[1],score_dict,mode)

    def _write_logger(self,epoch,model_type,score_dict,mode):
        self.logger=logging.getLogger('{}'.format(mode))
        if score_dict['custom_loss']==0.0:
            self.logger.info('\n[{} Epoch {}] [loss] {:.5f} [acc] {:.2f} [precision] {:.2f} [recall] {:.2f} [f1score] {:.2f}'.format(
                epoch,model_type,score_dict['loss'], score_dict['accuracy'],score_dict['precision'],score_dict['recall'],score_dict['f1score']))
        else:
            self.logger.info('\n[{} Epoch {}] [ce loss] {:.5f} [custom loss] {:.5f} [acc] {:.2f} [precision] {:.2f} [recall] {:.2f} [f1score] {:.2f}'.format(
                epoch,model_type,score_dict['loss'],score_dict['custom_loss'], score_dict['accuracy'],score_dict['precision'],score_dict['recall'],score_dict['f1score']))
        self.logWriter.add_scalars('{}_{}'.format(mode,model_type),score_dict,epoch)
    

class CrimeLearner(TorchLearner):
    def __init__(self,logger,imte_data, data_path, save_path, device, configs):
        super(CrimeLearner,self).__init__(logger,imte_data, data_path, save_path, device, configs)
        
class PriorityLearner(TorchLearner):
    def __init__(self,logger,imte_data, data_path, save_path, device, configs):
        super(PriorityLearner,self).__init__(logger,imte_data, data_path, save_path, device, configs)
    
