import torch

def calc_score(outputs,targets,score_dict):
    # this_correct=torch.eq(torch.max(outputs,dim=1)[1],targets).sum()
    bsz=targets.size(0)
    prediction=torch.max(outputs,dim=1)[1]
    TP = (targets * prediction).sum().to(torch.float)
    TN = ((1 - targets) * (1 - prediction)).sum().to(torch.float)
    FP = ((1 - targets) * prediction).sum().to(torch.float)
    FN = (targets * (1 - prediction)).sum().to(torch.float)
    epsilon=1e-7
    score_dict['total']+=bsz
    score_dict['TP']+=TP.item()
    score_dict['FP']+=FP.item()
    score_dict['FN']+=FN.item()
    score_dict['TN']+=TN.item()
    score_dict['accuracy'] = ((score_dict['TP']+score_dict['TN'])/score_dict['total']*100.0)
    score_dict['precision'] = (score_dict['TP']/(score_dict['TP']+score_dict['FP']+epsilon)*100.0)
    score_dict['recall'] = (score_dict['TP']/(score_dict['TP']+score_dict['FN']+epsilon)*100.0)
        
    score_dict['f1score']=(2*score_dict['precision']*score_dict['recall'])/(score_dict['precision']+score_dict['recall']+epsilon)
    
    return score_dict