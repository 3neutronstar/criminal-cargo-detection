#a = np.load('./data/custom_contest/train_data.npy')
#print(a.shape)
import pandas as pd
import numpy as np

df = pd.read_csv('./data/custom_contest/train.csv')
print(df.head())

def preprocessing(df):
    df = df.drop(columns=['신고번호', '신고일자'])#, '검사결과코드', ])#'우범여부', '핵심적발'])
    
    print(df.keys())
    for key in df.keys():
        df[key].replace(np.nan,"nan", inplace=True)
    categorical_cutting_features = ['신고인부호', '수입자부호', '해외거래처부호','반입보세구역부호','HS10단위부호']#, '관세율_categorical']
    categorical_features = ['통관지세관부호', '특송업체부호', '관세율_categorical',
                            '수입통관계획코드', '수입신고구분코드', '수입거래구분코드', '수입종류코드', '징수형태코드',
                            '운송수단유형코드', '반입보세구역부호', '적출국가코드', '원산지국가코드', '관세율구분코드']
    numeric_features = ['신고중량(KG)', '과세가격원화금액']#, '관세율']
    # 관세율, HS10단위부호는 별개

    if True: # 관세율
        df['관세율_categorical'] = df['관세율'].map(str)



    label = df['우범여부']
    for column in categorical_cutting_features:
        submit = df[column]
        submit_np, label_np = np.array(submit).reshape(len(submit), 1), np.array(label).reshape(len(submit), 1)
        # 우범여부 masking
        total_label_idx = np.where(label_np>=0)
        fraud_label_idx = np.where(label_np==1)
        total = submit_np[total_label_idx[0]]
        only_fraud = submit_np[fraud_label_idx[0]]
        # 높은 우범횟수 순으로 정렬
        elem_t, count_total = np.unique(total, return_counts=True)
        elem_f, count_fraud = np.unique(only_fraud, return_counts=True)
        submit_dict = dict(zip(elem_t, count_total))
        for i in range(len(elem_f)):
            total = submit_dict[elem_f[i]]
            submit_dict[elem_f[i]] = np.array((total, count_fraud[i]))
        for key in submit_dict:
            if type(submit_dict[key]) != np.ndarray:
                submit_dict[key] = np.array((submit_dict[key], np.int64(0)))
        # 우범횟수의 지분 낮은것 cutting
        if column in ['신고인부호', '해외거래처부호', 'HS10단위부호']:
            cnt = 10#10
        else:        #'수입자부호'
            cnt = 20#15
        filtered_count = filter(lambda x: x[1][0] >= cnt, submit_dict.items())
        # 비율이 아주낮거나 높으면 살림
        #filtered_ratio = filter(lambda x: x[1][0] <= cnt and x[1][0] >= 6 and (x[1][1]/x[1][0] <= 0.15 or x[1][1]/x[1][0] >= 0.7), submit_dict.items())
                                
        #filtered_features = np.array(list(filtered_ratio)+list(filtered_count))[:,0]
        filtered_features = np.array(list(filtered_count))[:,0]
        
        if column == 'HS10단위부호':
            df[column] = df[column].map(lambda x: x if x in filtered_features else 0)
            #df[column] = df[column].map(str)
        else:
            df[column] = df[column].map(lambda x: x if x in filtered_features else 'rare')


        print(f'{column} : {len(df[column].unique())}')

    if True: # HS10단위부호
        df['HS10단위부호_12'] = df['HS10단위부호'].map(lambda x: str(x//100000000))
        df['HS10단위부호_34'] = df['HS10단위부호'].map(lambda x: str((x//1000000)%100))
        #df['HS10단위부호_56'] = df['HS10단위부호'].map(lambda x: str((x//10000)%100))
        df = df.drop(columns=['HS10단위부호'])
        pass

    for column in categorical_features:
        df[column] = df[column].map(str)

    #df = df.drop(columns=['신고인부호','해외거래처부호', '수입자부호', '반입보세구역부호'])

    for column in numeric_features:
        df[column] = df[column].map(lambda x: np.log10(x+1))

    print(df.head())
    data_one_hot = pd.get_dummies(df)
    print(data_one_hot.head())

    

    return data_one_hot

data = preprocessing(df)

data2 = data.drop(columns=['우범여부', '핵심적발'])
print(data2.tail())

table = np.array(data2)
print(table.shape)

np.save('train_data.npy', table)

print(1)

# python main.py train_mixed --epoch 20