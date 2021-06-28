import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
# 2x2 matrix
def calc_score_2x2(predictions,targets,score_dict):
    bsz=targets.size(0)

    TP = (targets * predictions).sum().to(torch.float)
    TN = ((1 - targets) * (1 - predictions)).sum().to(torch.float)
    FP = ((1 - targets) * predictions).sum().to(torch.float)
    FN = (targets * (1 - predictions)).sum().to(torch.float)
    epsilon=1e-7
    score_dict['total']+=bsz
    score_dict['TP']+=TP.item()
    score_dict['FP']+=FP.item()
    score_dict['FN']+=FN.item()
    score_dict['TN']+=TN.item()
    score_dict['accuracy'] = ((score_dict['TP']+score_dict['TN'])/(score_dict['total'])*100.0)
    score_dict['precision'] = (score_dict['TP']/(score_dict['TP']+score_dict['FP']+epsilon)*100.0)
    score_dict['recall'] = (score_dict['TP']/(score_dict['TP']+score_dict['FN']+epsilon)*100.0)
        
    score_dict['f1score']=(2*score_dict['precision']*score_dict['recall'])/(score_dict['precision']+score_dict['recall']+epsilon)
    
    return score_dict


def calc_score(predictions, targets,score_dict):
    # confusion = confusion_matrix(targets, predictions)
    predictions=predictions.detach().clone().cpu()
    targets=targets.detach().clone().cpu()
    # print(targets, predictions)
    accuracy = accuracy_score(targets, predictions,)
    precision = precision_score(targets, predictions,average='weighted',zero_division=0)
    recall = recall_score(targets, predictions,average='weighted',zero_division=0)
    f1 = f1_score(targets, predictions,average='weighted',zero_division=0)
    score_dict['total']+=targets.size(0)
    score_dict['accuracy'] = accuracy*100.0
    score_dict['precision'] = precision*100.0
    score_dict['recall'] = recall*100.0
    score_dict['f1score']=f1*100.0
    print_dict=classification_report(targets,predictions,zero_division=0,digits=4)
    print(print_dict)
    return score_dict

# def calc_score(predictions,targets,score_dict):
#     score_dict['total']+=targets.size(0)
#     report=classification_report(targets,predictions)
#     score_dict.update(report)
