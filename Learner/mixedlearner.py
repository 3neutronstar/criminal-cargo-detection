from Learner.indlearner import TorchLearner
import torch
import copy
class MixedLearner(TorchLearner):
    def __init__(self, logger,time_data, data_path, save_path, device, configs):
        super(MixedLearner,self).__init__(logger,time_data, data_path, save_path, device, configs)
        self.score_dict={
            'loss':0.0,
            'total':0.0,
            'crime':copy.deepcopy(self.score_dict),
            'priority':copy.deepcopy(self.score_dict),
        }
        self.best_f1score={'crime':0.0,'priority':0.0}
        self.best_acc={'crime':0.0,'priority':0.0}
    
    def _train(self,epoch,score_dict):
        self.model.train()
        train_loss=0.0

        for batch_idx, (crime_data, priority_data, crime_targets, priority_targets) in enumerate(self.train_dataloader):
            crime_data, priority_data = crime_data.to(self.configs['device']), priority_data.to(self.configs['device'])
            crime_targets, priority_targets = crime_targets.to(self.configs['device']), priority_targets.to(self.configs['device'])
            crime_outputs,priority_outputs=self.model(crime_data, priority_data)
            crime_loss,priority_loss=self.criterion(crime_outputs,priority_outputs,crime_targets,priority_targets)
            loss=crime_loss+priority_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            crime_predictions=torch.max(crime_outputs,dim=1)[1].clone()
            priority_predictions=torch.max(priority_outputs,dim=1)[1].clone()

            #loss
            score_dict['crime']['loss']+=crime_loss.item()
            score_dict['priority']['loss']+=priority_loss.item()
            priority_predictions[crime_predictions==0]=-1
            self._save_score(crime_predictions,crime_targets,priority_predictions+1,priority_targets)
            train_loss+=loss.item()
            score_dict['total']+=crime_targets.size(0)
            if batch_idx%50==1:
                crime_acc=(self.metric['crime']['predictions']==self.metric['crime']['targets']).sum()/self.metric['crime']['targets'].size(0)*100.0
                priority_acc=(self.metric['priority']['predictions']==self.metric['priority']['targets']).sum()/self.metric['priority']['targets'].size(0)*100.0
                print('\r{}epoch {}/{}, [Crime Acc] {:.2f} [Priority Acc] {:.2f}  [Loss] {:.5f}'.format(epoch,int(score_dict['total']),
                len(self.train_dataloader.dataset),crime_acc,priority_acc,train_loss/(batch_idx+1)),end='')

        # crime
        score_dict['crime']=self._get_score(self.metric['crime']['predictions'],self.metric['crime']['targets'],score_dict['crime'])
        # priority
        score_dict['priority']=self._get_score(self.metric['priority']['predictions'],self.metric['priority']['targets'],score_dict['priority'])

        score_dict['loss']=train_loss/(batch_idx+1)
        score_dict['crime']['loss']=score_dict['crime']['loss']/(batch_idx+1)
        score_dict['priority']['loss']=score_dict['priority']['loss']/(batch_idx+1)
        return score_dict


    def _eval(self,epoch,score_dict):

        self.model.eval()
        eval_loss=0.0

        with torch.no_grad():
            for batch_idx,(crime_data,priority_data,crime_targets,priority_targets) in enumerate(self.test_dataloader):
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
                self._save_score(crime_predictions,crime_targets,priority_predictions+1,priority_targets)
                eval_loss +=loss.item()
        # crime
        score_dict['crime']=self._get_score(self.metric['crime']['predictions'],self.metric['crime']['targets'],score_dict['crime'])
        # priority
        score_dict['priority']=self._get_score(self.metric['priority']['predictions'],self.metric['priority']['targets'],score_dict['priority'])
                
        score_dict['loss']=eval_loss/(batch_idx+1)
        score_dict['crime']['loss']=score_dict['crime']['loss']/(batch_idx+1)
        score_dict['priority']['loss']=score_dict['priority']['loss']/(batch_idx+1)

        return score_dict

    def _save_score(self,crime_predictions,crime_targets,priority_predictions,priority_targets):
        if self.metric==dict():
            self.metric['crime']=dict()
            self.metric['priority']=dict()
            self.metric['crime']['predictions']= crime_predictions
            self.metric['crime']['targets']= crime_targets
            self.metric['priority']['predictions']= priority_predictions
            self.metric['priority']['targets']= priority_targets
        else:
            self.metric['crime']['predictions']=torch.cat((self.metric['crime']['predictions'],crime_predictions),dim=0)
            self.metric['crime']['targets']=torch.cat((self.metric['crime']['targets'],crime_targets),dim=0)
            self.metric['priority']['predictions']=torch.cat((self.metric['priority']['predictions'],priority_predictions),dim=0)
            self.metric['priority']['targets']=torch.cat((self.metric['priority']['targets'],priority_targets),dim=0)
            
    def _epoch_end_logger(self,epoch,score_dict,mode='train'):
        
        for model_type in ['crime','priority']:
            this_score_dict=score_dict[model_type]
            self._write_logger(epoch,model_type,this_score_dict,mode)
        if mode=='eval':
            for model_type in ['crime','priority']:
                if self.best_f1score[model_type]<score_dict[model_type]['f1score']:
                    self.best_f1score[model_type]=score_dict[model_type]['f1score']
                    self.best_acc[model_type]=score_dict[model_type]['accuracy']
                    self.save_models(epoch,score_dict,model_type)
    
    def load_model(self):
        crime_dict=torch.load(self.save_path,self.configs['file_name'],'best_crime_model.pt')
        priority_dict=torch.load(self.save_path,self.configs['file_name'],'best_priority_model.pt')
        dict_model={'crime':crime_dict,'priority':priority_dict}
        self.model.load_model(dict_model)
        