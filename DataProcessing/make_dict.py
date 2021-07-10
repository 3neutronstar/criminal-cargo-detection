import numpy as np

class MappingJsonGenerator():
    def __init__(self, train_csv, test_csv, fillna_str, drop_list):
        
        self.fillna_str = fillna_str
        self.drop_list = drop_list
        self.crime = np.array(train_csv['우범여부'])
        self.priority = np.array(train_csv['핵심적발'])
        self.train_csv=train_csv.drop(['우범여부', '핵심적발'] + self.drop_list, axis = 1,errors='ignore')
        self.test_csv=test_csv.drop(self.drop_list, axis = 1,errors='ignore')
        
        self.train_csv['HS_upper'] = self.train_csv['HS10단위부호'] // 100000000
        self.train_csv['HS_middle'] = self.train_csv['HS10단위부호'] // 1000000
        self.train_csv['HS_low'] = self.train_csv['HS10단위부호'] // 10000

        self.test_csv['HS_upper'] = self.test_csv['HS10단위부호'] // 100000000
        self.test_csv['HS_middle'] = self.test_csv['HS10단위부호'] // 1000000
        self.test_csv['HS_low'] = self.test_csv['HS10단위부호'] // 10000

        self.train_csv['관세율구분코드_1자리']=self.train_csv['관세율구분코드'].str.slice(start = 0, stop = 1)
        self.test_csv['관세율구분코드_1자리']=self.test_csv['관세율구분코드'].str.slice(start = 0, stop = 1)
        self.train_csv.drop('관세율구분코드',axis=1,inplace=True)
        self.test_csv.drop('관세율구분코드',axis=1,inplace=True)
        # self.train_csv = self.train_csv.drop(['HS10단위부호'], axis = 1)
        # self.test_csv = self.test_csv.drop(['HS10단위부호'], axis = 1)

        # self.train_hs_upper_code = np.array([s // 100000000 for s in self.train_hs_code]).reshape(-1, 1)
        # self.test_hs_upper_code = np.array([s // 100000000 for s in self.test_hs_code]).reshape(-1, 1)

        # self.train_hs_middle_code = np.array([s // 1000000 for s in self.train_hs_code]).reshape(-1, 1)
        # self.test_hs_middle_code = np.array([s // 1000000 for s in self.test_hs_code]).reshape(-1, 1)

        self.train_csv = self.train_csv.fillna(self.fillna_str)
        self.test_csv = self.test_csv.fillna(self.fillna_str)
        self.column_list = np.array(self.train_csv.columns, dtype=str)
        self.crime_idx = np.where(self.crime == 1)[0]
        self.non_priority_idx = np.where(self.priority == 1)[0]
        self.priority_idx = np.where(self.priority == 2)[0]
        self.dictionary = dict()
        
    def __call__(self):
        return self.forward()
    
    def forward(self):
        train_np, test_np = np.array(self.train_csv, dtype = str), np.array(self.test_csv, dtype = str)

        print(train_np.shape,test_np.shape)
        for i, col in enumerate(self.column_list):
            self.dictionary[col] = {}
            concat = np.concatenate([train_np[:, i], test_np[:, i]], axis = 0)
            total_code, total_count = np.unique(concat, return_counts=True)
            crime_code, crime_count = np.unique(train_np[:, i][self.crime_idx], return_counts=True)

            _, total_priority_count = np.unique(np.concatenate((train_np[:, i][self.priority_idx],train_np[:, i][self.non_priority_idx]),axis=0), return_counts=True)
            priority_code, priority_count = np.unique(train_np[:, i][self.priority_idx], return_counts=True)

            crime_ratio = np.empty((total_count.shape[0], ))
            
            priority_ratio = np.empty((total_count.shape[0], ))

            c_idx = 0
            p_idx = 0
            for assign_idx, c in enumerate(total_code) : 
                if c not in crime_code : 
                    crime_ratio[assign_idx] = 0.
                else :
                    crime_ratio[assign_idx] = np.round(crime_count[c_idx] / total_count[assign_idx], 4)
                    c_idx += 1

                self.dictionary[col][c] = {}
                self.dictionary[col][c]['total_count'] = int(total_count[assign_idx])
                self.dictionary[col][c]['crime_count'] = int(crime_count[c_idx-1])
                self.dictionary[col][c]['crime_ratio'] = float(crime_ratio[assign_idx])
                self.dictionary[col][c]['onehot'] = int(assign_idx)
            
            crime_concat = np.concatenate([crime_ratio.reshape(-1, 1), total_code.reshape(-1, 1)], axis = 1)
            crime_concat = np.array(sorted(crime_concat, key = lambda x : x[0], reverse=True))

            for i, c in enumerate(crime_concat[:, 1]):
                self.dictionary[col][c]['sorted_crime_onehot'] = int(i)
                
            for assign_idx,p in enumerate(total_code):
                if p not in priority_code: 
                    priority_ratio[assign_idx] = 0.
                else :
                    assign_priority_idx=np.where(priority_code==p)[0]
                    priority_ratio[assign_idx] = np.round(priority_count[p_idx] / total_priority_count[assign_priority_idx], 4)
                    p_idx += 1
                self.dictionary[col][p]['priority_ratio'] = float(priority_ratio[assign_idx])

            priority_concat = np.concatenate([priority_ratio.reshape(-1, 1), total_code.reshape(-1, 1)], axis = 1)
            priority_concat = np.array(sorted(priority_concat, key = lambda x : x[0], reverse=True))

            for i, c in enumerate(priority_concat[:, 1]):
                self.dictionary[col][c]['sorted_priority_onehot'] = int(i)

        return self.dictionary