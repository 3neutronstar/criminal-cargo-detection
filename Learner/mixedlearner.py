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
        self.best_f1score=[0.0,0.0]
        self.best_acc=[0.0,0.0]
    
    def _train(self,epoch,score_dict):
        self.model.train()
        train_loss=0.0

        for batch_idx, (data,crime_targets,priority_targets) in enumerate(self.train_dataloader):
            data,crime_targets,priority_targets=data.to(self.configs['device']),crime_targets.to(self.configs['device']),priority_targets.to(self.configs['device'])

            crime_outputs,priority_outputs=self.model(data)
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
            train_loss+=loss.item()
            # crime
            score_dict['crime']=self._get_score(crime_predictions,crime_targets,score_dict['crime'])
            # priority
            priority_predictions[crime_predictions==0]=-1
            priority_targets[crime_predictions==0]=-1
            score_dict['priority']=self._get_score(priority_predictions+1,priority_targets+1,score_dict['priority'])
            score_dict['total']+=crime_targets.size(0)
            if batch_idx%50==1:
                print('\r{}epoch {}/{}, [Crime Acc] {:.2f} [Priority Acc] {:.2f}  [Loss] {:.5f}'.format(epoch,int(score_dict['total']),
                len(self.train_dataloader.dataset),score_dict['crime']['accuracy'],score_dict['priority']['accuracy'],train_loss/(batch_idx+1)),end='')

        score_dict['loss']=train_loss/(batch_idx+1)
        score_dict['crime']['loss']=score_dict['crime']['loss']/(batch_idx+1)
        score_dict['priority']['loss']=score_dict['priority']['loss']/(batch_idx+1)
        return score_dict

  
    def _eval(self,epoch,score_dict):

        self.model.eval()
        eval_loss=0.0

        with torch.no_grad():
            for batch_idx,(data,crime_targets,priority_targets) in enumerate(self.test_dataloader):
                data,crime_targets,priority_targets=data.to(self.configs['device']),crime_targets.to(self.configs['device']),priority_targets.to(self.configs['device'])

                crime_outputs,priority_outputs=self.model(data)
                crime_loss,priority_loss=self.criterion(crime_outputs,priority_outputs,crime_targets,priority_targets)
                crime_predictions=torch.max(crime_outputs,dim=1)[1].clone()
                priority_predictions=torch.max(priority_outputs,dim=1)[1].clone()

                #loss
                loss=crime_loss+priority_loss
                score_dict['crime']['loss']+=crime_loss.item()
                score_dict['priority']['loss']+=priority_loss.item()
                # crime
                score_dict['crime']=self._get_score(crime_predictions,crime_targets,score_dict['crime'])
                # priority
                priority_predictions[crime_predictions==0]=-1
                priority_targets[crime_predictions==0]=-1
                score_dict['priority']=self._get_score(priority_predictions+1,priority_targets+1,score_dict['priority'])
                eval_loss +=loss.item()

        score_dict['loss']=eval_loss/(batch_idx+1)
        score_dict['crime']['loss']=score_dict['crime']['loss']/(batch_idx+1)
        score_dict['priority']['loss']=score_dict['priority']['loss']/(batch_idx+1)

        return score_dict       