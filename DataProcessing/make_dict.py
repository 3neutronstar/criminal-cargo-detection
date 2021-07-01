import numpy as np

class make_dict():
    def __init__(self, train_csv, test_csv):
        train_csv = train_csv
        test_csv = test_csv
        self.crime = np.array(train_csv['우범여부'])
        train_csv = train_csv.drop(['우범여부', '핵심적발', '신고번호', '신고일자', '신고중량(KG)', '과세가격원화금액', '관세율', '검사결과코드'], axis = 1)
        test_csv = test_csv.drop(['신고번호', '신고일자', '신고중량(KG)', '과세가격원화금액', '관세율', '검사결과코드'], axis = 1)
        self.train_csv = train_csv.fillna('Missing')
        self.test_csv = test_csv.fillna('Missing')
        self.column_list = np.array(self.train_csv.columns, dtype=str)
        self.crime_idx = np.where(self.crime == 1)[0]
        self.dictionary = dict()
        
    def __call__(self):
        return self.forward()
    
    def forward(self):
        
        train_np, test_np = np.array(self.train_csv, dtype = str), np.array(self.test_csv, dtype = str)
        
        for i in range(len(self.column_list)):
            self.dictionary[self.column_list[i]] = {}
            concat = np.concatenate([train_np[:, i], test_np[:, i]], axis = 0)
            total_code, total_count = np.unique(concat, return_counts=True)
            crime_code, crime_count = np.unique(train_np[:, i][self.crime_idx], return_counts=True)
            crime_ratio = np.empty((total_count.shape[0], ))

            c_idx = 0
            for assign_idx, c in enumerate(total_code) : 
                if c not in crime_code : 
                    crime_ratio[assign_idx] = 0.
                else :
                    crime_ratio[assign_idx] = np.round(crime_count[c_idx] / total_count[assign_idx], 4)
                    c_idx += 1

                self.dictionary[self.column_list[i]][c] = {}
                self.dictionary[self.column_list[i]][c]['count'] = int(total_count[assign_idx])
                self.dictionary[self.column_list[i]][c]['ratio'] = float(crime_ratio[assign_idx])
                self.dictionary[self.column_list[i]][c]['onehot'] = int(assign_idx)
                
        return self.dictionary