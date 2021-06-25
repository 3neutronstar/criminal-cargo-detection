
import time
import torch
import logging
from DataProcessing.load_data import load_dataloader
from Utils.calc_score import calc_score
from Model.basemodel import MODEL
import copy
class BaseLearner:
    def __init__(self,logger, datapath, savepath, device, configs):
        self.datapath=datapath
        self.savepath=savepath
        self.device=device
        self.train_dataloader,self.test_dataloader=load_dataloader(self.datapath,configs)
        self.configs=configs
        self.logger=logger
        input_space=self.train_dataloader.dataset[0][0].size()[0]
        output_space=2
        self.model=MODEL[configs['mode'].split('_')[1]](input_space,output_space,configs)

        self.criterion=self.model.criterion
        self.optimizer=self.model.optimizer
        self.scheduler=self.model.scheduler

    def run(self):
        self.model.to(self.configs['device'])
        score_dict={
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
        best_f1score=0.0
        best_acc=0.0
        self.logger.info(self.configs)

        for epoch in range(1,self.configs['epochs']+1):
            train_score_dict=copy.deepcopy(score_dict)
            eval_score_dict=copy.deepcopy(score_dict)
            #Init

            #Train
            print('='*30)
            train_tik=time.time()
            train_score_dict=self._train(epoch,train_score_dict)
            train_tok=time.time()
            print('\n Learning Rate: {:.8f} Learning Time: {:.3f}s'.format(self.optimizer.param_groups['lr'],train_tok-train_tik))
            self.logger=logging.getLogger('train')
            self.logger.info('\n[{}Epoch] [loss] {:.5f} [acc] {:.2f} [precision] {:.2f} [recall] {:.2f} [f1score] {:.2f}'.format(
                epoch,train_score_dict['loss'], train_score_dict['accuracy'],train_score_dict['precision'],train_score_dict['recall'],train_score_dict['f1score']))
            
            #Eval
            eval_score_dict=self._eval(epoch,eval_score_dict)
            self.logger=logging.getLogger('eval')
            self.logger.info('\n[{}Epoch] [loss] {:.5f} [acc] {:.2f} [precision] {:.2f} [recall] {:.2f} [f1score] {:.2f}'.format(
                epoch,eval_score_dict['loss'], eval_score_dict['accuracy'],eval_score_dict['precision'],eval_score_dict['recall'],eval_score_dict['f1score']))

            if best_f1score<eval_score_dict['f1score']:
                best_f1score=eval_score_dict['f1score']
                best_acc=eval_score_dict['accuracy']

            self.scheduler.step()

        self.logger = logging.getLogger('best')
        self.logger.info('[Mode {}] [Best Acc {:.2f}] [Best F1 {:.3f}]'.format(self.configs['mode'],best_acc,best_f1score))
        print('==End==')

    def _train(self,epoch,score_dict):
        self.model.train()
        train_loss=0.0

        epoch_total=0.0
        for batch_idx, (data,targets) in enumerate(self.train_dataloader):
            data,targets=data.to(self.configs['device']),targets.to(self.configs['device'])

            outputs=self.model(data)
            loss=self.criterion(outputs,targets)
            score_dict=calc_score(outputs,targets,score_dict)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss+=loss.item()
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
                score_dict=calc_score(outputs,targets,score_dict)
                loss=self.criterion(outputs,targets)

                eval_loss +=loss.item()
        score_dict['loss']=eval_loss/(batch_idx+1)

        return score_dict