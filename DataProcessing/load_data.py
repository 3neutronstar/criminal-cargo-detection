import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch
def load_npy(data_path,configs):
    npy_dict=None
    if configs['preprocess']==False:
        # ms, sj
        table_data=np.load(os.path.join(data_path,'mod_dh_data.npy'))
        table_data=np.load(os.path.join(data_path,'mod_data.npy'))
        crime_target=np.load(os.path.join(data_path,'mod_crime_target.npy')) # y_1 (0,1)
        priority_target=np.load(os.path.join(data_path,'mod_priority_target.npy')) #y_2 (0,1,2)
        train_priority_indices=np.load(os.path.join(data_path,'mod_train_priority_index.npy'))
        test_priority_indices=np.load(os.path.join(data_path,'mod_test_priority_index.npy'))
        train_crime_indices=np.load(os.path.join(data_path,'mod_train_crime_index.npy'))
        test_crime_indices=np.load(os.path.join(data_path,'mod_test_crime_index.npy'))
    else:
        # from DataProcessing.preprocessing import get_data
        from DataProcessing.gen_data import get_data
        npy_dict=get_data(data_path,configs)
    if 'xgboost' not in configs['mode']:
        table_data=torch.tensor(  table_data,dtype=torch.float)
        crime_target=torch.tensor(crime_target)
        priority_target=torch.tensor(priority_target)
        train_crime_indices=torch.tensor(train_crime_indices)
        test_crime_indices=torch.tensor(test_crime_indices)
        train_priority_indices=torch.tensor(train_priority_indices)
        test_priority_indices=torch.tensor(test_priority_indices)
    if npy_dict==None:
        npy_dict={
            'table_data':table_data,
            'crime_target':crime_target,
            'priority_target':priority_target,
            'train_crime_indices':train_crime_indices,
            'test_crime_indices':test_crime_indices,
            'train_priority_indices':train_priority_indices,
            'test_priority_indices':test_priority_indices,
        }
    return npy_dict


def load_dataset(data_path,configs):
    npy_dict=load_npy(data_path,configs)

    if 'xgboost' not in configs['mode']:
        if 'crime' in configs['mode']:
            crime_dataset=TensorDataset(npy_dict['table_data'],npy_dict['crime_target'])
            #crime
            train_dataset=Subset(crime_dataset,npy_dict['train_crime_indices'])
            test_dataset=Subset(crime_dataset,npy_dict['test_crime_indices'])

        elif 'priority' in configs['mode']:
            #priority
            train_data=npy_dict['table_data'][npy_dict['train_priority_indices'].long()]
            test_data=npy_dict['table_data'][npy_dict['test_priority_indices'].long()]

            train_target=npy_dict['priority_target'][npy_dict['train_priority_indices'].long()]
            test_target=npy_dict['priority_target'][npy_dict['test_priority_indices'].long()]

            # select 1 and 2
            train_data=train_data[torch.logical_or(train_target==1,train_target==2)]
            test_data=test_data[torch.logical_or(test_target==1,test_target==2)]
            train_target=train_target[torch.logical_or(train_target==1,train_target==2)]
            test_target=test_target[torch.logical_or(test_target==1,test_target==2)]

            # change target 1 to 0, 2 to 1
            train_target[train_target==1]=0
            train_target[train_target==2]=1
            test_target[test_target==1]=0
            test_target[test_target==2]=1
            train_dataset=TensorDataset(train_data,train_target)
            test_dataset=TensorDataset(test_data,test_target)

        elif configs['mode']=='train_mixed':
            #mixed
            mixed_dataset=TensorDataset(npy_dict['table_data'],npy_dict['crime_target'],npy_dict['priority_target'])
            train_dataset=Subset(mixed_dataset,npy_dict['train_priority_indices'])
            test_dataset=Subset(mixed_dataset,npy_dict['test_priority_indices'])

        else:
            print('No dataset')
            raise NotImplementedError
        return train_dataset, test_dataset
    else: #XGBOOST
        if 'crime' in configs['mode']:
            train_data=npy_dict['table_data'][npy_dict['train_crime_indices']]
            test_data=npy_dict['table_data'][npy_dict['test_crime_indices']]

            train_target=npy_dict['crime_target'][npy_dict['train_crime_indices']]
            test_target=npy_dict['crime_target'][npy_dict['test_crime_indices']]
        elif 'priority' in configs['mode']:
                
            train_data=npy_dict['table_data'][npy_dict['train_priority_indices']]
            test_data=npy_dict['table_data'][npy_dict['test_priority_indices']]

            train_target=npy_dict['priority_target'][npy_dict['train_priority_indices']]
            test_target=npy_dict['priority_target'][npy_dict['test_priority_indices']]
        return train_data,train_target,test_data,test_target
        

def load_dataloader(data_path,configs):
    if 'xgboost' not in configs['mode']:
        train_dataset,test_dataset=load_dataset(data_path,configs)
        train_dataloader=DataLoader(train_dataset,batch_size=configs['batch_size'],shuffle=True,)
        test_dataloader=DataLoader(test_dataset,batch_size=configs['batch_size'],shuffle=False)
        return train_dataloader,test_dataloader
    else: #xgboost
        train_data,train_target,test_data,test_target = load_dataset(data_path,configs)       
        return train_data,train_target,test_data,test_target
