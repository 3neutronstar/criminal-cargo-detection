import matplotlib.pyplot as plt
from DataProcessing.load_data import load_dataset
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import os
class Tsne:
    def __init__(self,data_path,save_path,current_path,configs):
        self.data_path=data_path
        self.save_path=save_path
        self.current_path=current_path
        self.configs=configs
        self.npy_dict=load_dataset(data_path,configs)

    def run(self):
        manifoldtsne=TSNE(n_components=2,verbose=1).fit_transform(self.npy_dict['table_data'])
        self._plt_crime(manifoldtsne)
        print("after shape:",manifoldtsne.shape)
        if self.configs['mode']=='tsne_crime':
            self._plt_crime(manifoldtsne)
        elif self.configs['mode']=='tsne_priority':
            self._plt_priority(manifoldtsne)
        else:
            self._plt_crime(manifoldtsne)
            self._plt_priority(manifoldtsne)
    
    def _plt_crime(self,manifoldtsne):
        tsne_df=pd.DataFrame({'x':manifoldtsne[:,0],'y':manifoldtsne[:,1],'classes':self.npy_dict['crime_targets']})
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x='x',y='y',
            hue='classes',
            data=tsne_df,
            legend='full',
            alpha=0.1
        )
        plt.savefig(os.path.join(self.save_path,'plt_crime.jpg'))
        plt.clf()
    
    def _plt_priority(self,manifoldtsne):
        tsne_df=pd.DataFrame({'x':manifoldtsne[:,0],'y':manifoldtsne[:,1],'classes':self.npy_dict['priority_targets']})
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x='x',y='y',
            hue='classes',
            data=tsne_df,
            legend='full',
            alpha=0.1
        )
        plt.savefig(os.path.join(self.save_path,'plt_priority.jpg'))
        plt.clf()




            



