import numpy as np
import sys

class MappingJsonGenerator():
    def __init__(self, train_csv, test_csv, fillna_str, drop_list):
        
        self.fillna_str = fillna_str
        self.drop_list = drop_list
        self.crime = np.array(train_csv['우범여부'])
        self.priority = np.array(train_csv['핵심적발'])
        self.train_csv = train_csv.drop(['우범여부', '핵심적발'] + self.drop_list, axis = 1)
        self.test_csv = test_csv.drop(self.drop_list, axis = 1)
        
        # Split HS10-code to HS_upper and HS_middle
        self.train_csv['HS_upper'] = self.train_csv['HS10단위부호'].astype(str).str.slice(start = 0, stop = 2)
        self.train_csv['HS_middle'] = self.train_csv['HS10단위부호'].astype(str).str.slice(start = 2, stop = 4)

        self.test_csv['HS_upper'] = self.test_csv['HS10단위부호'].astype(str).str.slice(start = 0, stop = 2)
        self.test_csv['HS_middle'] = self.test_csv['HS10단위부호'].astype(str).str.slice(start = 2, stop = 4)

        self.train_csv = self.train_csv.drop(['HS10단위부호'], axis = 1)
        self.test_csv = self.test_csv.drop(['HS10단위부호'], axis = 1)

        # Filling nan to 'Missing'
        self.train_csv = self.train_csv.fillna(self.fillna_str)
        self.test_csv = self.test_csv.fillna(self.fillna_str)

        self.column_list = np.array(self.train_csv.columns, dtype=str)
        self.crime_idx = np.where(self.crime == 1)[0]
        self.priority_idx = np.where(self.priority == 2)[0]
        self.crime_proportion = self.crime_idx.shape[0]/self.crime.shape[0]
        self.priority_proportion = self.priority_idx.shape[0]/self.priority.shape[0]
        self.dictionary = dict()
        
    def __call__(self):
        return self.forward()
    
    def forward(self):
        
        train_np, test_np = np.array(self.train_csv, dtype = str), np.array(self.test_csv, dtype = str)
        print(train_np.shape,test_np.shape)

        for i in range(len(self.column_list)):
            
            self.dictionary[str(self.column_list[i])] = {}
            concat = np.concatenate([train_np[:, i], test_np[:, i]], axis = 0)
            total_code, total_count = np.unique(concat, return_counts=True)
            train_code, _ = np.unique(train_np[:, i], return_counts = True)
            test_code, _ = np.unique(test_np[:, i], return_counts = True)
            train_crime_code, train_crime_count = np.unique(train_np[:, i][self.crime_idx], return_counts=True)
            train_priority_code, train_priority_count = np.unique(train_np[:, i][self.priority_idx], return_counts=True)

            n_ratio = 0.

            for assign_idx, c in enumerate(total_code) : 
                
                # Make key, value pair in dictionary
                self.dictionary[self.column_list[i]][c] = {}

                total_c = total_count[assign_idx]

                # In case of, code is only in test.csv
                if c not in train_code : 
                    self.dictionary[self.column_list[i]][c]['total_count'] = int(total_c)
                    self.dictionary[self.column_list[i]][c]['crime_count'] = int(0)
                    self.dictionary[self.column_list[i]][c]['priority_count'] = int(0)
                    self.dictionary[self.column_list[i]][c]['crime_ratio'] = float(0.)
                    self.dictionary[self.column_list[i]][c]['priority_ratio'] = float(0.)
                    self.dictionary[self.column_list[i]][c]['is_mask'] = True

                # In case of, code is only in test.csv
                elif c not in test_code :
                    self.dictionary[self.column_list[i]][c]['total_count'] = int(total_c)
                    self.dictionary[self.column_list[i]][c]['crime_count'] = int(0)
                    self.dictionary[self.column_list[i]][c]['priority_count'] = int(0)
                    self.dictionary[self.column_list[i]][c]['crime_ratio'] = float(0.)
                    self.dictionary[self.column_list[i]][c]['priority_ratio'] = float(0.)
                    self.dictionary[self.column_list[i]][c]['is_mask'] = True

                # In case of, code is in train.csv
                else :
                    where_crime_code = np.where(train_crime_code == c)[0]
                    where_priority_code = np.where(train_priority_code == c)[0]

                    if (len(where_crime_code) ==  0) and (len(where_priority_code) == 0): 
                        crime_c = 0
                        priority_c = 0
                        crime_ratio = 0.
                        priority_ratio = 0.

                    elif (len(where_crime_code) ==  0) and not (len(where_priority_code) == 0): 
                        crime_c = 0
                        priority_c = train_priority_count[where_priority_code][0]
                        crime_ratio = 0.
                        priority_ratio = (priority_c / total_c)

                    elif not (len(where_crime_code) ==  0) and (len(where_priority_code) == 0): 
                        crime_c = train_crime_count[where_crime_code][0]
                        priority_c = 0
                        crime_ratio = (crime_c / total_c)
                        priority_ratio = 0.

                    else :
                        crime_c = train_crime_count[where_crime_code][0]
                        priority_c = train_priority_count[where_priority_code][0]
                        crime_ratio = (crime_c / total_c)
                        priority_ratio = (priority_c / total_c)

                    # Compute absolute substraction of crime-proportion and standard
                    proportion = np.abs((crime_c / total_c) - self.crime_proportion)

                    #if proportion < 0.05 :
                    if False :
                        self.dictionary[self.column_list[i]][c]['total_count'] = int(total_c)
                        self.dictionary[self.column_list[i]][c]['crime_count'] = int(0)
                        self.dictionary[self.column_list[i]][c]['priority_count'] = int(0)
                        self.dictionary[self.column_list[i]][c]['crime_ratio'] = float(0.)
                        self.dictionary[self.column_list[i]][c]['priority_ratio'] = float(0.)
                        self.dictionary[self.column_list[i]][c]['is_mask'] = True

                    else :
                        self.dictionary[self.column_list[i]][c]['total_count'] = int(total_c)
                        self.dictionary[self.column_list[i]][c]['crime_count'] = int(crime_c)
                        self.dictionary[self.column_list[i]][c]['priority_count'] = int(priority_c)
                        self.dictionary[self.column_list[i]][c]['crime_ratio'] = float(np.round(crime_ratio, 6))
                        self.dictionary[self.column_list[i]][c]['priority_ratio'] = float(np.round(priority_ratio, 6))
                        self.dictionary[self.column_list[i]][c]['is_mask'] = False

                    # Exeptional condition for error cases
                    if total_code[assign_idx] != c : 
                        print('Caused Error!')
                        sys.exit()
        
        return self.dictionary

# Example
# ret_dict = make_dict(train_csv, test_csv, 'Missing', ['신고일자', '신고중량(KG)', '과세가격원화금액', '관세율'])()