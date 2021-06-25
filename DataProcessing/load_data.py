import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data import Subset
import torch

def load_dataset(datapath,configs):
    table_data=torch.tensor(np.load(os.path.join(datapath,'mod_data.npy')))
    crime_target=torch.tensor(np.load(os.path.join(datapath,'mod_crime_target.npy')))
    priority_target=torch.tensor(np.load(os.path.join(datapath,'mod_priority_target.npy')))
    train_indices=torch.tensor(np.load(os.path.join(datapath,'mod_train_index.npy')))
    test_indices=torch.tensor(np.load(os.path.join(datapath,'mod_test_index.npy')))
   

    if configs['mode']=='train_crime':
        crime_dataset=TensorDataset(table_data,crime_target)
        #crime
        train_dataset=Subset(crime_dataset,train_indices)
        test_dataset=Subset(crime_dataset,test_indices)
    elif configs['mode']=='train_priority':
        #priority
        priority_dataset=TensorDataset(table_data,priority_target.float())
        train_dataset=Subset(priority_dataset,train_indices)
        test_dataset=Subset(priority_dataset,test_indices)
    elif configs['mode']=='train_mixed':
        #mixed
        mixed_dataset=TensorDataset(table_data,crime_target,priority_target)
        train_dataset=Subset(mixed_dataset,train_indices)
        test_dataset=Subset(mixed_dataset,test_indices)
    else:
        print('No dataset')
        raise NotImplementedError
    return train_dataset, test_dataset

def load_dataloader(datapath,configs):
    train_dataset,test_dataset=load_dataset(datapath,configs)
    train_dataloader=DataLoader(train_dataset,batch_size=configs['batch_size'],shuffle=True,)
    test_dataloader=DataLoader(test_dataset,batch_size=configs['batch_size'],shuffle=False)
    return train_dataloader,test_dataloader

