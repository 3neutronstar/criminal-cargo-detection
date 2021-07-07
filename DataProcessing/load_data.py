from DataProcessing.preprocessing import Preprocessing
import os
import sklearn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data import Subset
import torch
def load_dataset(data_path,configs):
    # get data
    if configs['preprocess']==True:
        # npy_dict=get_data(data_path,configs)
        preprocessing=Preprocessing(data_path,configs)
        npy_dict=preprocessing.run()
    if 'sj' in configs['mode']:
        train_data=np.load(os.path.join(data_path,'sj_train_data.npy'))
        train_targets=np.load(os.path.join(data_path,'sj_train_targets.npy'))
        valid_data=np.load(os.path.join(data_path,'sj_valid_data.npy'))
        valid_targets=np.load(os.path.join(data_path,'sj_valid_targets.npy'))

        train_dataset=TensorDataset(train_data,train_targets)
        valid_dataset=TensorDataset(valid_data,valid_targets)
        return train_dataset,valid_dataset
    else:    
        npy_dict={
            'table_data':np.load(os.path.join(data_path,'train_data.npy')),
            'crime_targets':np.load(os.path.join(data_path,'crime_targets.npy')), # y_1 (0,1)
            'priority_targets':np.load(os.path.join(data_path,'priority_targets.npy')), #y_2 (0,1,2)
            'train_indices':np.load(os.path.join(data_path,'train_indices.npy')),
            'valid_indices':np.load(os.path.join(data_path,'valid_indices.npy')),
            'test_data':np.load(os.path.join(data_path,'test_data.npy')),
        }
    
    if configs['mode']=='record':
        return npy_dict
    #type casting
    if 'xgboost' not in configs['mode']:
        for key in npy_dict.keys():
            npy_dict[key]=torch.from_numpy(npy_dict[key])
            if npy_dict[key].dtype in [torch.float64,torch.double]:
                npy_dict[key]=npy_dict[key].float()

    #data separation
        if 'crime' in configs['mode']:
            crime_dataset=TensorDataset(npy_dict['table_data'],npy_dict['crime_targets'])
            #crime
            train_dataset=Subset(crime_dataset,npy_dict['train_indices'])
            valid_dataset=Subset(crime_dataset,npy_dict['valid_indices'])

        elif 'priority' in configs['mode']:
            #priority
            train_data=npy_dict['table_data'][npy_dict['train_indices'].long()]
            valid_data=npy_dict['table_data'][npy_dict['valid_indices'].long()]

            train_target=npy_dict['priority_targets'][npy_dict['train_indices'].long()]
            valid_target=npy_dict['priority_targets'][npy_dict['valid_indices'].long()]

            # select 1 and 2
            train_data=train_data[torch.logical_or(train_target==1,train_target==2)]
            valid_data=valid_data[torch.logical_or(valid_target==1,valid_target==2)]
            train_target=train_target[torch.logical_or(train_target==1,train_target==2)]
            valid_target=valid_target[torch.logical_or(valid_target==1,valid_target==2)]

            # change target 1 to 0, 2 to 1
            train_target[train_target==1]=0
            train_target[train_target==2]=1
            valid_target[valid_target==1]=0
            valid_target[valid_target==2]=1
            train_dataset=TensorDataset(train_data,train_target)
            valid_dataset=TensorDataset(valid_data,valid_target)

        elif configs['mode']=='train_mixed':
            #mixed
            mixed_dataset=TensorDataset(npy_dict['table_data'],npy_dict['crime_targets'],npy_dict['priority_targets'])
            train_dataset=Subset(mixed_dataset,npy_dict['train_indices'])
            valid_dataset=Subset(mixed_dataset,npy_dict['valid_indices'])

        else:
            print('No dataset')
            raise NotImplementedError
        return train_dataset, valid_dataset

    else: #XGBOOST
        if 'crime' in configs['mode']:
            train_data=npy_dict['table_data'][npy_dict['train_indices']]
            valid_data=npy_dict['table_data'][npy_dict['valid_indices']]

            train_target=npy_dict['crime_targets'][npy_dict['train_indices']]
            valid_target=npy_dict['crime_targets'][npy_dict['valid_indices']]
        elif 'priority' in configs['mode']:
                
            train_data=npy_dict['table_data'][npy_dict['train_indices']]
            valid_data=npy_dict['table_data'][npy_dict['valid_indices']]

            train_target=npy_dict['priority_targets'][npy_dict['train_indices']]
            valid_target=npy_dict['priority_targets'][npy_dict['valid_indices']]
        return train_data,train_target,valid_data,valid_target
        

def load_dataloader(data_path,configs):
    if 'xgboost' not in configs['mode']:
        train_dataset,valid_dataset=load_dataset(data_path,configs)
        train_dataloader=DataLoader(train_dataset,batch_size=configs['batch_size'],shuffle=True,num_workers=configs['num_workers'])
        valid_dataloader=DataLoader(valid_dataset,batch_size=configs['batch_size'],shuffle=False,num_workers=configs['num_workers'])
        return train_dataloader,valid_dataloader
    else: #xgboost
        train_data,train_target,valid_data,valid_target = load_dataset(data_path,configs)       
        return train_data,train_target,valid_data,valid_target
