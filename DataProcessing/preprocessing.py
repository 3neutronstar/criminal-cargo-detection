# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import copy
import pandas as pd
import csv
import os
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame
from DataProcessing.make_dict import MappingJsonGenerator

def find_digits(x):
    temp = math.log2(x)
    if temp == math.floor(temp):
        n = temp
    else:
        n = math.floor(temp) + 1
    return int(n)

def binary_transform(x):
    if x == 0:
        return '0'
    else:
        binary=''
        while x>0:
            x, mod = divmod(x,2)
            binary += str(mod)
        return binary

class RescaleNumeric:
    def __init__(self):
        self.minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
    
    def __call__(self,x):
        x_scale=self.minmax_scale.fit_transform(np.log(x+1))
        return x_scale

class Preprocessing:
    def __init__(self,data_path,configs):
        self.configs=configs
        self.data_path=data_path
        # load mapping dictionary
        if os.path.exists(os.path.join(data_path,'mapping.json'))==False:
            train_dataframe,test_dataframe=self._load_dataset()
            self.mapping_dict=MappingJsonGenerator(train_dataframe,test_dataframe,'Missing', ['신고일자', '신고중량(KG)', '과세가격원화금액', '관세율'])()
            print("Generate Json complete")
            with open(os.path.join(data_path,'mapping.json'), 'w') as fp:
                json.dump(self.mapping_dict, fp, indent=2)
        else:
            with open(os.path.join(data_path,'mapping.json'), 'r') as fp:
                self.mapping_dict = json.load(fp)
        #

    def _load_dataset(self):
        train_dataframe=copy.deepcopy(pd.read_csv(os.path.join(self.data_path,'train.csv')))
        test_dataframe=copy.deepcopy(pd.read_csv(os.path.join(self.data_path,'test.csv')))
        return train_dataframe,test_dataframe
    
    def run(self) ->dict:
        npy_dict={}
        #load dataset
        train_dataframe,test_dataframe=self._load_dataset()
        if self.configs['mode']=='record':
            data_frame_list=[test_dataframe]
            data_type_list=['test']
        else:
            data_frame_list=[train_dataframe,test_dataframe]
            data_type_list=['train','test']

        #transform and save the dataset
        for csv_dataframe,data_type in zip(data_frame_list,data_type_list):
            if data_type=='train':
                npy_dict['crime_targets']=csv_dataframe.pop('우범여부')
                npy_dict['priority_targets']=csv_dataframe.pop('핵심적발')
                npy_dict['train_indices'], npy_dict['valid_indices']=self._split_indices(csv_dataframe,npy_dict['priority_targets'])
            npy_dict['{}_data'.format(data_type)]=self._transform(csv_dataframe,data_type)
        for key in npy_dict.keys():
            if isinstance(npy_dict,DataFrame):
                npy_dict[key]=npy_dict[key].to_numpy()
            np.save(os.path.join(self.data_path,'{}.npy'.format(key)),npy_dict[key])
        
        return npy_dict
    
    def _split_indices(self,dataframe:DataFrame,targets:np.ndarray)->np.ndarray:
        indices=np.arange(len(dataframe))
        train_indices,valid_indices=train_test_split(indices,stratify=targets,random_state=self.configs['seed'],test_size=1-self.configs['split_ratio'],train_size=self.configs['split_ratio'])
        return train_indices, valid_indices

    def _transform(self, dataframe:DataFrame, train_or_test)->DataFrame:
        rescaler=RescaleNumeric()
        """
        categorical_features = ['통관지세관부호', '신고인부호', '수입자부호', '해외거래처부호', '특송업체부호', 
                                '수입통관계획코드', '수입신고구분코드', '수입거래구분코드', '수입종류코드', 
                                '징수형태코드', '운송수단유형코드', '반입보세구역부호', 'HS10단위부호', 
                                '적출국가코드', '원산지국가코드', '관세율구분코드']
        """
        print("Before transform shape",dataframe.shape)
        categorical_features = self.mapping_dict.keys()
        numeric_features = ['신고중량(KG)', '과세가격원화금액']

        if train_or_test == 'train':
          dataframe.drop(['신고일자','신고번호','검사결과코드'],axis=1,inplace=True)
        elif train_or_test == 'test':
          dataframe.drop(['신고일자','검사결과코드'],axis=1,inplace=True)

        dataframe.fillna('Missing', inplace=True)

        for column in numeric_features:
            dataframe[column] = rescaler(np.log(dataframe.pop(column).to_numpy()+1).reshape(-1,1))

        np_data = dataframe[['신고중량(KG)', '과세가격원화금액','관세율']].to_numpy()
        len_df = len(dataframe.index)
        
        for i,column in enumerate(categorical_features):
            if column not in dataframe.columns:
                continue
            dataframe[column] = dataframe[column].map(str)
            dict_col = self.mapping_dict[column]
            np_count_ratio = np.zeros((len_df,2))

            max_ohe = len(dict_col.keys())
            encoding_digits = find_digits(max_ohe) 
            np_encoding = np.zeros((len_df,encoding_digits))

            for row in dataframe[column].index: 
                val_data = dataframe[column][row]
                np_count_ratio[row][0] = dict_col[val_data]['count']
                np_count_ratio[row][1] = dict_col[val_data]['ratio']

                x = binary_transform(dict_col[val_data]['onehot']) 
                len_x = len(x) 
                for idx in range(len_x): 
                  if x[idx]=='1':
                    np_encoding[row][idx] = x[idx]
                    
            np_count_ratio[:,0] = np_count_ratio[:,0]/(np_count_ratio[:,0].max())
            np_count_ratio[:,1] = np_count_ratio[:,1]/(np_count_ratio[:,1].max())
            np_encoding = np_encoding[:,::-1]

            np_concat = np.concatenate((np_count_ratio, np_encoding),axis=1)
            np_data = np.concatenate((np_data,np_concat), axis=1)
            
            print('\r[{}/{}] Finished Process'.format(i+1,len(categorical_features)),end='')
        print("After transform shape",np_data.shape)
        #print(np_data.columns)
        return np_data
