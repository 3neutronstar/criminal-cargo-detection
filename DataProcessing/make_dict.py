import numpy as np

class MappingJsonGenerator():
    def __init__(self, train_csv, test_csv, fillna_str, drop_list):
        
        self.fillna_str = fillna_str
        self.drop_list = drop_list
        self.crime = np.array(train_csv['우범여부'])
        self.priority = np.array(train_csv['핵심적발'])
        self.train_csv = train_csv.drop(['우범여부', '핵심적발'] + self.drop_list, axis = 1,errors='ignore')
        self.test_csv = test_csv.drop(self.drop_list, axis = 1,errors='ignore')
        
        self.train_hs_code = np.array(self.train_csv['HS10단위부호'], dtype = str)
        self.test_hs_code = np.array(self.test_csv['HS10단위부호'], dtype = str)
        
        self.train_csv = self.train_csv.drop(['HS10단위부호'], axis = 1)
        self.test_csv = self.test_csv.drop(['HS10단위부호'], axis = 1)
        
        self.train_hs_upper_code = np.array([s[:2] for s in self.train_hs_code]).reshape(-1, 1)
        self.test_hs_upper_code = np.array([s[:2] for s in self.test_hs_code]).reshape(-1, 1)

        self.train_hs_middle_code = np.array([s[2:4] for s in self.train_hs_code]).reshape(-1, 1)
        self.test_hs_middle_code = np.array([s[2:4] for s in self.test_hs_code]).reshape(-1, 1)

        self.train_csv = self.train_csv.fillna(self.fillna_str)
        self.test_csv = self.test_csv.fillna(self.fillna_str)
        self.column_list = np.array(np.concatenate([self.train_csv.columns, np.array(['HS_upper', 'HS_middle'], dtype = str)], axis = 0), dtype=str)
        self.crime_idx = np.where(self.crime == 1)[0]
        self.non_priority_idx = np.where(self.priority == 1)[0]
        self.priority_idx = np.where(self.priority == 2)[0]
        self.dictionary = dict()
        
    def __call__(self):
        return self.forward()
    
    def forward(self):
        train_np, test_np = np.array(self.train_csv, dtype = str), np.array(self.test_csv, dtype = str)
        train_np = np.concatenate([train_np, self.train_hs_upper_code, self.train_hs_middle_code], axis = 1)
        test_np = np.concatenate([test_np, self.test_hs_upper_code, self.test_hs_middle_code], axis = 1)
        print(train_np.shape,test_np.shape)
        for i, col in enumerate(self.column_list):
            self.dictionary[col] = {}
            concat = np.concatenate([train_np[:, i], test_np[:, i]], axis = 0)
            total_code, total_count = np.unique(concat, return_counts=True)
            crime_code, crime_count = np.unique(train_np[:, i][self.crime_idx], return_counts=True)

            total_priority_code, total_priority_count = np.unique(np.concatenate((train_np[:, i][self.priority_idx],train_np[:, i][self.non_priority_idx]),axis=0), return_counts=True)
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
                self.dictionary[col][c]['count'] = int(total_count[assign_idx])
                self.dictionary[col][c]['ratio'] = float(crime_ratio[assign_idx])
                self.dictionary[col][c]['onehot'] = int(assign_idx)
            for assign_idx,p in enumerate(total_code):
                if p not in priority_code: 
                    priority_ratio[assign_idx] = 0.
                else :
                    assign_priority_idx=np.where(priority_code==p)[0]
                    priority_ratio[assign_idx] = np.round(priority_count[p_idx] / total_priority_count[assign_priority_idx], 4)
                    p_idx += 1
                self.dictionary[col][p]['priority_ratio'] = float(priority_ratio[assign_idx])
                
        return self.dictionary