# -*- coding: utf-8 -*-
import math
import os, json, copy
import numpy as np
import pandas as pd
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
        x_scale=self.minmax_scale.fit_transform(np.log(x+1).reshape(-1,1))
        return x_scale


class Preprocessing:
    def __init__(self,data_path,configs):
        self.configs=configs
        self.data_path=data_path
        # load mapping dictionary
        train_dataframe,test_dataframe=self._load_dataset()
        npy_dict=dict()
        npy_dict['crime_targets']=train_dataframe.pop('우범여부')
        npy_dict['priority_targets']=train_dataframe.pop('핵심적발')
        npy_dict['train_indices'], npy_dict['valid_indices']=self._split_indices(train_dataframe,npy_dict['priority_targets'])
        for key in npy_dict.keys():
            if isinstance(npy_dict,DataFrame):
                npy_dict[key]=npy_dict[key].to_numpy()
            np.save(os.path.join(self.data_path,'{}.npy'.format(key)),npy_dict[key])
        del npy_dict
        train_dataframe,test_dataframe=self._load_dataset()
        self.mapping_dict = MappingJsonGenerator(train_dataframe,'Missing', ['신고번호', '신고일자', '관세율','검사결과코드'], self.configs['only_train'])() #'과세가격원화금액', '신고중량(KG)',
        print("Generate Json complete")
        # save
        with open(os.path.join(data_path,'mapping.json'), 'w') as fp:
            json.dump(self.mapping_dict, fp, indent=2)
        # load
        with open(os.path.join(data_path,'mapping.json'), 'r') as fp:
            self.mapping_dict = copy.deepcopy(json.load(fp))

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
            npy_dict['{}_priority_data'.format(data_type)]=self._priority_transform(copy.deepcopy(csv_dataframe))
            npy_dict['{}_crime_data'.format(data_type)]=self._crime_transform(copy.deepcopy(csv_dataframe))
        for key in npy_dict.keys():
            if isinstance(npy_dict,DataFrame):
                npy_dict[key]=npy_dict[key].to_numpy()
            np.save(os.path.join(self.data_path,'{}.npy'.format(key)),npy_dict[key])
        return npy_dict

    def _split_indices(self,dataframe:DataFrame,targets:np.ndarray)->np.ndarray:
        indices=np.arange(len(dataframe))
        train_indices,valid_indices=train_test_split(indices,stratify=targets,random_state=self.configs['seed'],test_size=1-self.configs['split_ratio'],train_size=self.configs['split_ratio'])
        return train_indices, valid_indices

    def _crime_transform(self, dataframe:DataFrame)->np.ndarray:
        rescaler=RescaleNumeric()
        """
        categorical_features = ['통관지세관부호', '신고인부호', '수입자부호', '해외거래처부호', '특송업체부호', 
                                '수입통관계획코드', '수입신고구분코드', '수입거래구분코드', '수입종류코드', 
                                '징수형태코드', '운송수단유형코드', '반입보세구역부호', 'HS_upper', 'HS_middle', 
                                '적출국가코드', '원산지국가코드', '관세율구분코드']
        """
        categorical_features = self.mapping_dict.keys()
        dataframe['HS_upper'] = dataframe['HS10단위부호'] // 100000000 # 위 2자리
        dataframe['HS_middle'] = dataframe['HS10단위부호'] // 1000000 # 위 4자리
        dataframe['HS_low'] = dataframe['HS10단위부호'] // 10000 # 위 4자리
        dataframe['관세율구분코드_1자리']=dataframe['관세율구분코드'].str.slice(start = 0, stop = 1)
        dataframe['단위무게(KG)가격'] = (dataframe['과세가격원화금액']/dataframe['신고중량(KG)']).map(lambda x: np.round(x, 0)).map(str)
        dataframe.drop('HS10단위부호',axis=1,inplace=True)
        # dataframe['세금'] = (dataframe['과세가격원화금액']*dataframe['관세율']/100.0)
        dataframe['세금'] = (dataframe['과세가격원화금액']*dataframe['관세율']/dataframe['신고중량(KG)']/100.0)
        # numeric_features = ['신고중량(KG)', '과세가격원화금액','관세율','세금']
        numeric_features = ['신고중량(KG)', '과세가격원화금액','관세율']
        dataframe.fillna('Missing', inplace=True)
        for column in numeric_features:
            dataframe[column] = rescaler(dataframe.pop(column).to_numpy())
        np_data = dataframe[['신고중량(KG)', '과세가격원화금액','관세율']].to_numpy()
        # numeric_features = ['신고중량(KG)', '과세가격원화금액','관세율']
        dataframe.drop(['신고일자','신고번호','우범여부','핵심적발' ,'수입자부호'],axis=1,inplace=True,errors='ignore')#,'HS10단위부호'
        len_df = len(dataframe.index)
        
        add_count_ratio_list=['crime_count','total_count']
        reg_count_ratio_list=['crime_count','total_count']
        np_encoding=None
        print("Before crime transform shape",dataframe.shape)
        for i,column in enumerate(categorical_features):
            if column not in dataframe.columns:
                continue
            dataframe[column] = dataframe[column].map(str)
            dict_col = self.mapping_dict[column]
            np_count_ratio = np.zeros((len_df,len(add_count_ratio_list)))
            if column in ['HS_upper','특송업체부호']:
                np_ratio_addition = np.zeros((len_df,1))
                for row in dataframe[column].index: 
                    val_data = dataframe[column][row]
                    if val_data not in dict_col.keys():
                        continue
                    else:
                        np_ratio_addition[row]=dict_col[val_data]['crime_ratio']
                np_count_ratio=np.concatenate((np_count_ratio,np_ratio_addition),axis=1)
            for row in dataframe[column].index: 
                val_data = dataframe[column][row]
                # value you want to add
                for idx, add_instance in enumerate(add_count_ratio_list):
                    if val_data not in dict_col.keys():
                        if add_instance in ['crime_ratio','priority_ratio'] : np_count_ratio[row][idx] = 0.0
                        else : np_count_ratio[row][idx] = 0
                    else :
                        if (not dict_col[val_data]['is_crime_mask']) and (dict_col[val_data]['crime_ratio'] >= 0.3) :
                            if add_instance != 'total_count':
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]*1.5
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]*1.5
                            else :
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                        elif (not dict_col[val_data]['is_crime_mask']) and (dict_col[val_data]['crime_ratio'] < 0.3) :
                            np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                            np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                        elif (dict_col[val_data]['is_crime_mask']) :
                            if add_instance != 'total_count':
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]*0.5
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]*0.5
                            else :
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                        else :
                            np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                            np_count_ratio[row][idx] = dict_col[val_data][add_instance]
            # # OneHot
            # max_ohe = len(dict_col.keys())+2
            # encoding_digits = find_digits(max_ohe) 
            # np_encoding = np.zeros((len_df,encoding_digits))
            # for row in dataframe[column].index: 
            #     val_data = dataframe[column][row]
            #     for idx, add_instance in enumerate(add_count_ratio_list):
            #         if val_data not in dict_col.keys():
            #             x = binary_transform(0) 
            #         else :
            #             if (not dict_col[val_data]['is_crime_mask']) and (dict_col[val_data]['crime_ratio'] >= 0.3) :
            #                 x = binary_transform(dict_col[val_data]['onehot']) 
            #             elif (not dict_col[val_data]['is_crime_mask']) and (dict_col[val_data]['crime_ratio'] < 0.3) :
            #                 x = binary_transform(dict_col[val_data]['onehot']) 
            #             elif (dict_col[val_data]['is_crime_mask']):
            #                 x = binary_transform(0) 
            #             else :
            #                 x = binary_transform(dict_col[val_data]['onehot']) 

            #     len_x = len(x) 
            #     if x=='0':
            #         for idx in range(len_x):
            #             np_encoding[row][idx] = 0.5
            #     else:
            #         for idx in range(len_x):
            #             np_encoding[row][idx] = float(x[idx])

            # regularization
            for idx,reg_instance in enumerate(reg_count_ratio_list):
                np_count_ratio[:,idx] = (np_count_ratio[:,idx]-np_count_ratio[:,idx].mean())/(np_count_ratio[:,idx].var())
            
            if np_encoding is None: # 이진화 안하는 경우 ->위 주석처리 
                np_data = np.concatenate((np_data,np_count_ratio, ), axis=1)
            else: #이진화 하는 경우 -> 위 주석처리 x
                np_encoding = np_encoding[:,::-1]
                np_data = np.concatenate((np_data,np_count_ratio,np_encoding ), axis=1)

            print('\r[{}/{}] Finished Process'.format(i+1,len(categorical_features)),end='')
                
        print("After crime transform shape",np_data.shape)
        return np_data

    def _priority_transform(self, dataframe:DataFrame)->np.ndarray:
        rescaler=RescaleNumeric()
        """
        categorical_features = ['통관지세관부호', '신고인부호', '수입자부호', '해외거래처부호', '특송업체부호', 
                                '수입통관계획코드', '수입신고구분코드', '수입거래구분코드', '수입종류코드', 
                                '징수형태코드', '운송수단유형코드', '반입보세구역부호', 'HS_upper', 'HS_middle', 
                                '적출국가코드', '원산지국가코드', '관세율구분코드']
        """
        dataframe.fillna('Missing', inplace=True)
        categorical_features = self.mapping_dict.keys()
        dataframe['HS_upper'] = dataframe['HS10단위부호'] // 100000000 # 위 2자리
        dataframe['HS_middle'] = dataframe['HS10단위부호'] // 1000000 # 위 4자리
        dataframe['HS_low'] = dataframe['HS10단위부호'] // 10000 # 위 4자리
        dataframe.drop('HS10단위부호',axis=1,inplace=True)
        dataframe['관세율구분코드_1자리']=dataframe['관세율구분코드'].str.slice(start = 0, stop = 1)
        dataframe['단위무게(KG)가격'] = (dataframe['과세가격원화금액']/dataframe['신고중량(KG)']).map(lambda x: np.round(x, 0)).map(str)
        dataframe['세금'] = (dataframe['과세가격원화금액']*dataframe['관세율']/dataframe['신고중량(KG)']/100.0)
        numeric_features = ['신고중량(KG)', '과세가격원화금액','관세율','세금']
        # numeric_features = ['신고중량(KG)', '과세가격원화금액','관세율']
        for column in numeric_features:
            dataframe[column] = rescaler(dataframe.pop(column).to_numpy())
        np_data = dataframe[['신고중량(KG)', '과세가격원화금액','관세율','세금']].to_numpy()
        np_encoding=None
        dataframe.drop(['신고일자','신고번호','우범여부','핵심적발', '수입자부호'],axis=1,inplace=True,errors='ignore')#,'HS10단위부호'
        len_df = len(dataframe.index)
        add_count_ratio_list=['priority_count','crime_count']
        reg_count_ratio_list=['priority_count','crime_count']

        print("Before priority transform shape",dataframe.shape)
        for i,column in enumerate(categorical_features):
            if column not in dataframe.columns:
                continue
            dataframe[column] = dataframe[column].map(str)
            dict_col = self.mapping_dict[column]
            np_count_ratio = np.zeros((len_df,len(add_count_ratio_list)))
            if column in ['HS_upper','특송업체부호']:
                np_ratio_addition = np.zeros((len_df,1))
                for row in dataframe[column].index: 
                    val_data = dataframe[column][row]
                    if val_data not in dict_col.keys():
                        continue
                    else:
                        np_ratio_addition[row]=dict_col[val_data]['priority_ratio']
                np_count_ratio=np.concatenate((np_count_ratio,np_ratio_addition),axis=1)
            for row in dataframe[column].index: 
                val_data = dataframe[column][row]

                # value you want to add
                for idx, add_instance in enumerate(add_count_ratio_list):
                    if val_data not in dict_col.keys():
                        if add_instance in ['crime_ratio','priority_ratio'] : np_count_ratio[row][idx] = 0.0
                        else : np_count_ratio[row][idx] = 0
                    else :
                        if (not dict_col[val_data]['is_priority_mask']) and (dict_col[val_data]['priority_ratio'] >= 0.3) :
                            if add_instance != 'crime_count':
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]*1.5
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]*1.5
                            else :
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                        elif (not dict_col[val_data]['is_priority_mask']) and (dict_col[val_data]['priority_ratio'] < 0.3) :
                            np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                            np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                        elif (dict_col[val_data]['is_priority_mask']) :
                            if add_instance != 'crime_count':
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]*0.5
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]*0.5
                            else :
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                                np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                        else :
                            np_count_ratio[row][idx] = dict_col[val_data][add_instance]
                            np_count_ratio[row][idx] = dict_col[val_data][add_instance]
            # # OneHot
            # max_ohe = len(dict_col.keys())+2
            # encoding_digits = find_digits(max_ohe) 
            # np_encoding = np.zeros((len_df,encoding_digits))
            # for row in dataframe[column].index: 
            #     val_data = dataframe[column][row]
            #     for idx, add_instance in enumerate(add_count_ratio_list):
            #         if val_data not in dict_col.keys():
            #             x = binary_transform(0) 
            #         else :
            #             if (not dict_col[val_data]['is_priority_mask']) and (dict_col[val_data]['priority_ratio'] >= 0.3) :
            #                 x = binary_transform(dict_col[val_data]['onehot']) 
            #             elif (not dict_col[val_data]['is_priority_mask']) and (dict_col[val_data]['priority_ratio'] < 0.3) :
            #                 x = binary_transform(dict_col[val_data]['onehot']) 
            #             elif (dict_col[val_data]['is_priority_mask']):
            #                 x = binary_transform(0) 
            #             else :
            #                 x = binary_transform(dict_col[val_data]['onehot']) 

            #     len_x = len(x) 
            #     if x=='0':
            #         for idx in range(len_x):
            #             np_encoding[row][idx] = 0.5
            #     else:
            #         for idx in range(len_x):
            #             np_encoding[row][idx] = float(x[idx])

            # regularization
            for idx,reg_instance in enumerate(reg_count_ratio_list):
                np_count_ratio[:,idx] = (np_count_ratio[:,idx]-np_count_ratio[:,idx].mean())/(np_count_ratio[:,idx].var())
            
            if np_encoding is None: # 이진화 안하는 경우 ->위 주석처리 
                np_data = np.concatenate((np_data,np_count_ratio, ), axis=1)
            else: #이진화 하는 경우 -> 위 주석처리 x
                np_encoding = np_encoding[:,::-1]
                np_data = np.concatenate((np_data,np_count_ratio,np_encoding ), axis=1)

            print('\r[{}/{}] Finished Process'.format(i+1,len(categorical_features)),end='')
                
        print("After priority transform shape",np_data.shape)
        return np_data
        
        """
        for i, column in enumerate(categorical_features):
            dataframe[column] = dataframe[column].map(str)
            dict_col = self.mapping_dict[column]
            np_count_ratio = np.zeros((len_df,3))

            for row in dataframe[column].index: 
                val_data = dataframe[column][row]  
                np_count_ratio[row][0] = dict_col[val_data]['crime_count']
                np_count_ratio[row][1] = dict_col[val_data]['crime_ratio']
                np_count_ratio[row][2] = dict_col[val_data]['priority_ratio']
                np_column = dataframe[column].to_numpy()
                if dict_col[val_data]['is_mask']==True:
                    np_column[row] = 'masking'
                    
            np_ohe = pd.get_dummies(np_column).to_numpy()
            np_count_ratio[:,0] = (np_count_ratio[:,0]-np_count_ratio[:,0].mean())/(np_count_ratio[:,0].var())
            
            np_concat = np.concatenate((np_count_ratio,np_ohe), axis=1)
            np_data = np.concatenate((np_data,np_concat), axis=1)
            print('\r[{}/{}] Finished Process'.format(i+1,len(categorical_features)),end='')
            print(f' [ {column: <9}\t] : {len(np.unique(np_column[:]))}, {np_column[:].dtype}')
        """
