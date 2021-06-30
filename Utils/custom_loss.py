import torch
import torch.nn.functional as F
import torch.nn as nn

def loss_kd_regularization(outputs, labels):
    """
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    alpha = 0.6
    T = 20.0
    correct_prob = 0.99    # the probability for correct class in u(k)
    loss_CE = F.cross_entropy(outputs, labels)
    K = outputs.size(1)

    teacher_soft = torch.ones_like(outputs)
    teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
    for i in range(outputs.shape[0]):
        teacher_soft[i ,labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1))*100

    KD_loss = (1. - alpha)*loss_CE + alpha*loss_soft_regu

    return KD_loss

def f_beta_score_loss(y_pred, y_true):
    assert y_pred.ndim == 2
    assert y_true.ndim == 1
    beta=1/10.0
    epsilon=1e-7
    y_true = F.one_hot(y_true, 2).to(torch.float32)
    y_pred = F.softmax(y_pred, dim=1)
    
    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = (1+beta**2)* (precision*recall) / (beta**2*precision + recall + 1e-7)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    return 1 - f1.mean()