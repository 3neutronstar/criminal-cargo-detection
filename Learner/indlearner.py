from Model.relmodel import MixedModel
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
        'TP':0.0,
        'TN':0.0,
        'FP':0.0,
        'FN':0.0,
        'accuracy':0.0,
        'total':0.0,
        'precision':0.0,
        'recall':0.0,
        'f1score':0.0,
        'loss':0.0,
        }

    def run(self):
        self.model.to(self.configs['device'])
        best_f1score=0.0
        best_acc=0.0
        self.logger.info(self.configs)

        for epoch in range(1,self.configs['epochs']+1):
            train_score_dict=copy.deepcopy(self.score_dict)
            eval_score_dict=copy.deepcopy(self.score_dict)
            #Init

            #Train
            print('='*30)
            train_tik=time.time()
            train_score_dict=self._train(epoch,train_score_dict)
            train_tok=time.time()
            print('\n Learning Rate: {:.8f} Learning Time: {:.3f}s'.format(self.optimizer.param_groups[0]['lr'],train_tok-train_tik))
            self._epoch_end_logger(epoch,train_score_dict,'train')

            #Eval
            eval_score_dict=self._eval(epoch,eval_score_dict)
            self._epoch_end_logger(epoch,train_score_dict,'eval')
            if best_f1score<eval_score_dict['f1score']:
                best_f1score=eval_score_dict['f1score']
                best_acc=eval_score_dict['accuracy']
                self.save_models(epoch,eval_score_dict)

            self.scheduler.step()

        self.logger = logging.getLogger('best')
        self.logger.info('[Mode {}] [Best Acc {:.2f}] [Best F1 {:.3f}]'.format(self.configs['mode'],best_acc,best_f1score))
        print('==End==')

    def _train(self,epoch,score_dict):
        self.model.train()
        train_loss=0.0

        for batch_idx, (data,targets) in enumerate(self.train_dataloader):
            data,targets=data.to(self.configs['device']),targets.to(self.configs['device'])

            outputs=self.model(data)
            loss=self.criterion(outputs,targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss+=loss.item()
            score_dict=self._get_score(outputs,targets,score_dict)
            if batch_idx%50==1:
                print('\r{}epoch {}/{}, [Acc] {:.2f} [Loss] {:.5f}'.format(epoch,int(score_dict['total']),
                len(self.train_dataloader.dataset),score_dict['accuracy'],train_loss/(batch_idx+1)),end='')
        score_dict['loss']=train_loss/(batch_idx+1)
        return score_dict

  
    def _eval(self,epoch,score_dict):

        self.model.eval()
        eval_loss=0.0

        with torch.no_grad():
            for batch_idx,(data,targets) in enumerate(self.test_dataloader):
                data,targets=data.to(self.configs['device']),targets.to(self.configs['device'])

                outputs=self.model(data)
                self._get_score(outputs,targets,score_dict)

                loss=self.criterion(outputs,targets)

                eval_loss +=loss.item()
        score_dict['loss']=eval_loss/(batch_idx+1)

        return score_dict

    def save_models(self,epoch, score_dict):
        dict_model={
            'epoch':epoch,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
        }.update(score_dict)
        torch.save(dict_model,
        os.path.join(self.save_path,self.time_data,'best_model.pt'))
    
    def _get_score(self,outputs,targets,score_dict):
        predictions=torch.max(outputs,dim=1)[1].clone()
        score_dict=calc_score(predictions,targets,score_dict)
        return score_dict
    
    def _epoch_end_logger(self,epoch,score_dict,mode='train'):
        if 'mixed' in self.configs['mode']:
            for model_type in ['crime','priority']:
                this_score_dict=score_dict[model_type]
                self._write_logger(epoch,model_type,this_score_dict,mode)
        else:
            self._write_logger(epoch,self.configs['mode'].split('_')[1],score_dict,mode)

    def _write_logger(self,epoch,model_type,score_dict,mode):
        self.logger=logging.getLogger('{}'.format(mode))
        self.logger.info('\n[{} Epoch {}] [loss] {:.5f} [acc] {:.2f} [precision] {:.2f} [recall] {:.2f} [f1score] {:.2f}'.format(
            epoch,model_type,score_dict['loss'], score_dict['accuracy'],score_dict['precision'],score_dict['recall'],score_dict['f1score']))
    


class CrimeLearner(TorchLearner):
    def __init__(self,logger,imte_data, data_path, save_path, device, configs):
        super(CrimeLearner,self).__init__(logger,imte_data, data_path, save_path, device, configs)
        
class PriorityLearner(TorchLearner):
    def __init__(self,logger,imte_data, data_path, save_path, device, configs):
        super(PriorityLearner,self).__init__(logger,imte_data, data_path, save_path, device, configs)
    
