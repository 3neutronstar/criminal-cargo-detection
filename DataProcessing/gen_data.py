import os
import pandas as pd
import torch
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
def gen_data(data_path):
    train_origin_data=pd.read_csv(os.path.join(data_path,'train.csv'))
    '''
    0은 바꿀 필요가 있음 o는 숫자이므로 유지
    신고번호 = x
    신고일자 = x
    통관지세관부호 = o
    신고인부호 = 0
    수입자부호 = 0
    해외 거래처 부호 = 0
    특송업체부호 = 0 

    '''
    train_origin_data=train_origin_data.fillna('missing')
    # 데이터 확인
    print('Data 종류')
    print(train_origin_data.shape)
    print(train_origin_data.columns)
    # 쓸모없는 데이터 날리기
    train_origin_data.drop('신고번호',axis=1,inplace=True)
    train_origin_data.drop('신고일자',axis=1,inplace=True)
    train_origin_data.drop('수입자부호',axis=1,inplace=True)
    train_origin_data.drop('검사결과코드',axis=1,inplace=True)
    # train_origin_data.drop('해외거래처부호',axis=1,inplace=True)
    # train_origin_data.drop('HS10단위부호',axis=1,inplace=True)

    # target 두개 분리
    crime_target=torch.tensor(train_origin_data.pop('우범여부').to_numpy())#,dtype=torch.float)
    priority_target=torch.tensor(train_origin_data.pop('핵심적발').to_numpy())
    #numerical data 분리
    train_weight=np.log(train_origin_data.pop('신고중량(KG)').to_numpy()+1).reshape(-1,1)
    train_price=np.log(train_origin_data.pop('과세가격원화금액').to_numpy()+1).reshape(-1,1)
    train_custom_rate=train_origin_data.pop('관세율').to_numpy().reshape(-1,1)

    #replace data
    train_submit=np.load(os.path.join(data_path,'submit.npy'),allow_pickle=True)
    train_express=np.load(os.path.join(data_path,'express.npy'),allow_pickle=True)
    train_import=np.load(os.path.join(data_path,'import.npy'),allow_pickle=True)
    train_company=np.load(os.path.join(data_path,'company.npy'),allow_pickle=True)
    train_origin_data['신고인부호']=train_submit
    train_origin_data['특송부호']=train_express
    train_origin_data['수입자부호']=train_import
    train_origin_data['해외업체부호']=train_company
    # 분리 확인
    print('제거 후 데이터 종류',train_origin_data.columns)
    print(train_origin_data.tail())

    for key in train_origin_data.keys():
        enc=OneHotEncoder().fit(train_origin_data[key].to_numpy().reshape(-1,1))
        encoded_data=enc.transform(train_origin_data[key].to_numpy().reshape(-1,1))
        print(key,':',encoded_data.shape)

    # One hot encoding
    enc=OneHotEncoder(dtype=np.float32).fit(train_origin_data.to_numpy().reshape(-1,len(train_origin_data.columns)))
    train_encoded_data=enc.transform(train_origin_data.to_numpy().reshape(-1,len(train_origin_data.columns))).toarray()
    print("encoded dataset",train_encoded_data.shape)

    # numerical dataset
    train_price_tensor=torch.tensor(train_price,dtype=torch.float)
    train_weight_tensor=torch.tensor(train_weight,dtype=torch.float)
    train_custom_rate_tensor=torch.tensor(train_custom_rate,dtype=torch.float)
    # categorical dataset -> encoded data
    train_encoded_data_tensor=torch.tensor(train_encoded_data,dtype=torch.float)
    train_tensor_data=torch.cat((train_encoded_data_tensor,train_price_tensor,train_weight_tensor,train_custom_rate_tensor),dim=1)


    indices=np.arange(len(train_tensor_data))
    # train_tensor_data=torch.cat((train_price_tensor,train_weight_tensor,train_custom_rate_tensor),dim=1)
    del train_price,train_weight,train_custom_rate,train_encoded_data
    print(train_tensor_data.size())
    train_crime_indices,test_crime_indices=train_test_split(indices,stratify=crime_target)
    train_priority_indices,test_priority_indices=train_test_split(indices,stratify=priority_target)

    np.save(os.path.join(data_path,'mod_data.npy'),train_tensor_data.numpy())
    np.save(os.path.join(data_path,'mod_crime_target.npy'),crime_target)
    np.save(os.path.join(data_path,'mod_priority_target.npy'),priority_target)
    np.save(os.path.join(data_path,'mod_train_index.npy'),train_crime_indices)
    np.save(os.path.join(data_path,'mod_test_index.npy'),test_crime_indices)
    np.save(os.path.join(data_path,'mod_train_index.npy'),train_priority_indices)
    np.save(os.path.join(data_path,'mod_test_index.npy'),test_priority_indices)
    print('Transform Finished')
    return