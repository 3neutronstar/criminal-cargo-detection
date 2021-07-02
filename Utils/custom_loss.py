import torch
import torch.nn.functional as F
import torch.nn as nn

class KDRegLoss():
    def __init__(self,configs):
        self.configs=configs
        self.T=self.configs['temperature']# 20
        self.alpha=self.configs['alpha']# 0.6
    def __call__(self,outputs,labels):

        """
        loss function for mannually-designed regularization: Tf-KD_{reg}
        """
        self.alpha = 0.6
        self.T = 20.0
        correct_prob = 0.99    # the probability for correct class in u(k)
        K = outputs.size(1)

        teacher_soft = torch.ones_like(outputs)
        teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
        for i in range(outputs.shape[0]):
            teacher_soft[i ,labels[i]] = correct_prob
        loss_soft_regu = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/self.T, dim=1))*100

        KD_loss = self.alpha*loss_soft_regu

        return KD_loss

class FBetaLoss():
    def __init__(self,configs):
        self.configs=configs
        self.gae=self.configs['lambda']# 1.0
        self.beta=self.configs['beta']# 10

    def __call__(self,y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        epsilon=1e-7
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32) 

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = (1+self.beta**2)* (precision*recall) / (self.beta**2*precision + recall + 1e-7)
        f1 = f1.clamp(min=epsilon, max=1-epsilon)
        return self.gae*(1 - f1.mean())