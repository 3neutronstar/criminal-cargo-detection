import os

import pandas as pd
from DataProcessing.make_dict import MappingJsonGenerator
class Preprocessing:
    def __init__(self,data_path,configs):
        self.configs=configs
        if os.path.exists(os.path.join(data_path,'mapping.json'))==False:
            train_csv=pd.read_csv(os.path.join(data_path,'train.csv'))
            test_csv=pd.read_csv(os.path.join(data_path,'test.csv'))
            mapping_dict=MappingJsonGenerator(train_csv,test_csv)
            self.mapping_dict=mapping_dict.forward()

    
    def run(self):
        npy_dict={}
        
        return npy_dict