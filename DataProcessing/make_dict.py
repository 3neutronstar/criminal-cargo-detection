import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager, rc

class MappingJsonGenerator():
    def __init__(self, train_csv, fillna_str, drop_list, only_train, data_path):
        
        self.only_train = only_train
        self.train_csv=train_csv
        self.fillna_str = fillna_str
        self.drop_list = drop_list
        
        self.train_csv['HS_upper'] = self.train_csv['HS10단위부호'] // 100000000
        self.train_csv['HS_middle'] = self.train_csv['HS10단위부호'] // 1000000
        self.train_csv['HS_low'] = self.train_csv['HS10단위부호'] // 10000

        self.train_csv['관세율구분코드_1자리']=self.train_csv['관세율구분코드'].str.slice(start = 0, stop = 1)

        self.train_csv['단위무게(KG)가격'] = (self.train_csv['과세가격원화금액']/self.train_csv['신고중량(KG)']).map(lambda x: np.round(x, 0)).map(str)

        self.train_csv.drop(['과세가격원화금액', '신고중량(KG)'], axis = 1,errors='ignore',inplace=True)
        
        if self.only_train:
            # train_indices = np.load('./data/custom_contest/train_indices.npy')
            train_indices = np.load(os.path.join(data_path,'train_indices.npy'))
            self.train_valid_csv = self.train_csv
            self.train_csv = self.train_valid_csv[self.train_valid_csv.columns].iloc[train_indices]
            self.train_valid_csv=self.train_valid_csv.drop(['우범여부', '핵심적발'] + self.drop_list, axis = 1,errors='ignore')
            self.train_valid_csv = self.train_valid_csv.fillna(self.fillna_str)


        self.crime = np.array(self.train_csv['우범여부'])
        self.priority = np.array(self.train_csv['핵심적발'])
        self.train_csv=self.train_csv.drop(['우범여부', '핵심적발'] + self.drop_list, axis = 1,errors='ignore')

        self.train_csv = self.train_csv.fillna(self.fillna_str)
        self.column_list = np.array(self.train_csv.columns, dtype=str)

        self.crime_idx = np.where(self.crime == 1)[0]
        self.non_priority_idx = np.where(self.priority == 1)[0]
        self.priority_idx = np.where(self.priority == 2)[0]
        self.dictionary = dict()

        upper_limit = 0
        #self.crime_threshold_no_valid = [150, 7, 3, 3, 40, upper_limit, upper_limit, upper_limit, upper_limit, upper_limit, upper_limit, 20, 10, upper_limit, upper_limit, upper_limit, 30, 30, 20, upper_limit, 3]
        #self.priority_threshold_no_valid = [150, 16, 3, 3, 60, upper_limit, upper_limit, upper_limit, upper_limit, upper_limit, upper_limit, 30, 15, upper_limit, upper_limit, upper_limit, 30, 25, 15, upper_limit, 3]
        self.crime_threshold_no_valid = [0]*21
        self.priority_threshold_no_valid = [5]*21
        
    def __call__(self):
        return self.forward()
    
    def forward(self):
        train_np = np.array(self.train_csv, dtype = str)
        print(train_np.shape)
        if self.only_train:
            train_valid_np = np.array(self.train_valid_csv, dtype = str)
            print('calc count_ratio for only train')
            print(train_valid_np.shape)
        else:
            train_valid_np = train_np
            print('calc count_ratio for train + valid')
            print(train_np.shape)

        for i, col in enumerate(self.column_list):
            crime_count_mean, crime_ratio_mean = 0., 0.
            priority_count_mean, priority_ratio_mean = 0., 0.
            self.dictionary[col] = {}
            concat = np.concatenate([train_valid_np[:, i]], axis = 0)
            total_code, total_count = np.unique(concat, return_counts=True)
            crime_code, crime_count = np.unique(train_np[:, i][self.crime_idx], return_counts=True)

            _, total_priority_count = np.unique(np.concatenate((train_np[:, i][self.priority_idx],train_np[:, i][self.non_priority_idx]),axis=0), return_counts=True)
            priority_code, priority_count = np.unique(train_np[:, i][self.priority_idx], return_counts=True)
            
            #print(col)
            #--------------------------------------------
            # crime_dict = dict(zip(priority_code, priority_count))
            # crime_dict = np.array(sorted(crime_dict.items(), reverse=True, key = lambda x : x[1]))
            # #--------------------------------------------

            # font_path = "C:\\Windows\\Fonts\\gulim.ttc"
            # font = font_manager.FontProperties(fname=font_path).get_name()
            # rc('font', family=font)

            # plt.plot(crime_dict[:, 0], crime_dict[:, 1].astype(np.int))
            # plt.title(col+'/'+str(crime_dict[:, 0].shape[0]), fontsize = 20)
            # ax = plt.gca()
            # ax.axes.xaxis.set_visible(False)
            # plt.show()

            crime_ratio = np.empty((total_count.shape[0], ))
            
            priority_ratio = np.empty((total_count.shape[0], ))

            c_idx = 0
            p_idx = 0
            for assign_idx, c in enumerate(total_code) : 
                self.dictionary[col][c] = {}
                self.dictionary[col][c]['total_count'] = int(total_count[assign_idx])

                if c not in crime_code : 
                    crime_ratio[assign_idx] = 0.
                    self.dictionary[col][c]['crime_count'] = int(0)
                    self.dictionary[col][c]['is_crime_mask'] = True
                else :
                    crime_ratio[assign_idx] = np.round(crime_count[c_idx] / total_count[assign_idx], 4)
                    self.dictionary[col][c]['crime_count'] = int(crime_count[c_idx])
                    crime_count_mean += float(crime_count[c_idx])
                    crime_ratio_mean += float(crime_ratio[assign_idx]*crime_count[c_idx])
                    if crime_count[c_idx] <= self.crime_threshold_no_valid[i]:
                        self.dictionary[col][c]['is_crime_mask'] = True
                    else : 
                        self.dictionary[col][c]['is_crime_mask'] = False
                    c_idx += 1

                self.dictionary[col][c]['crime_ratio'] = float(crime_ratio[assign_idx])
                self.dictionary[col][c]['onehot'] = int(assign_idx+1)
            
            crime_concat = np.concatenate([crime_ratio.reshape(-1, 1), total_code.reshape(-1, 1)], axis = 1)
            crime_concat = np.array(sorted(crime_concat, key = lambda x : x[0], reverse=True))

            for k, c in enumerate(crime_concat[:, 1]):
                self.dictionary[col][c]['sorted_crime_onehot'] = int(k+1)


                
            for assign_idx,p in enumerate(total_code):
                if p not in priority_code: 
                    priority_ratio[assign_idx] = 0.
                    self.dictionary[col][p]['priority_count'] = int(0)
                    self.dictionary[col][p]['is_priority_mask'] = True
                else :
                    assign_priority_idx=np.where(priority_code==p)[0]
                    #priority_ratio[assign_idx] = np.round(priority_count[p_idx] / total_priority_count[assign_priority_idx], 4)
                    priority_ratio[assign_idx] = np.round(priority_count[p_idx] / self.dictionary[col][p]['crime_count'], 4)
                    self.dictionary[col][p]['priority_count'] = int(priority_count[p_idx])
                    priority_count_mean += float(priority_count[p_idx])
                    priority_ratio_mean += float(priority_ratio[assign_idx]*priority_count[p_idx])
                    if priority_count[p_idx] <= self.priority_threshold_no_valid[i]:
                        self.dictionary[col][p]['is_priority_mask'] = True
                    else : 
                        self.dictionary[col][p]['is_priority_mask'] = False
                    p_idx += 1

                
                self.dictionary[col][p]['priority_ratio'] = float(priority_ratio[assign_idx])

            priority_concat = np.concatenate([priority_ratio.reshape(-1, 1), total_code.reshape(-1, 1)], axis = 1)
            priority_concat = np.array(sorted(priority_concat, key = lambda x : x[0], reverse=True))

            for k, c in enumerate(priority_concat[:, 1]):
                self.dictionary[col][c]['sorted_priority_onehot'] = int(k+1)

            self.dictionary[col]['crime_count_mean'] = crime_count_mean / train_valid_np.shape[0]
            self.dictionary[col]['crime_ratio_mean'] =crime_ratio_mean / train_valid_np.shape[0]
            self.dictionary[col]['priority_count_mean'] = priority_count_mean / train_valid_np.shape[0]
            self.dictionary[col]['priority_ratio_mean'] = priority_ratio_mean / train_valid_np.shape[0]

        return self.dictionary