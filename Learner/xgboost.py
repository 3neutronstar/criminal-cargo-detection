
from DataProcessing.load_data import load_dataloader
from Learner.baselearner import BaseLearner

from Learner.baselearner import BaseLearner
from Utils.calc_score import calc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix,precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, make_scorer

def get_clf_eval(y_test, pred=None, pred_proba=None):
  confusion = confusion_matrix(y_test, pred)
  accuracy = accuracy_score(y_test, pred)
  precision = precision_score(y_test, pred)
  recall = recall_score(y_test, pred)
  f1 = f1_score(y_test, pred)
  roc_acu = roc_auc_score(y_test, pred_proba)
  print('오차행렬')
  print(confusion)
  print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f},\nF1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_acu))


class XGBoostLearner(BaseLearner):
    def __init__(self,logger,time_data, datapath, savepath, device, configs):
        super(XGBoostLearner,self).__init__(logger,time_data, datapath, savepath, device, configs)
        self.train_data,self.train_target,self.test_data,self.test_target=load_dataloader(datapath,configs)
        
    def run(self):
        #  xgb_crime = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
        xgbmodel = XGBClassifier(objective='binary:logistic', missing=1)

        # 학습
        xgbmodel.fit(self.train_data, self.train_target,
                    verbose=True,
                    early_stopping_rounds=100, 
                    eval_metric='error',
                    eval_set=[(self.test_data,self.test_target)])

        # 추론
        w_crime_preds = xgbmodel.predict(self.test_data)
        w_crime_pred_proba = xgbmodel.predict_proba(self.test_data)[:,1]
        get_clf_eval(self.test_target, w_crime_preds, w_crime_pred_proba)

        # confusion_matrix 시각화
        plot_confusion_matrix(xgbmodel,
                            self.test_data, self.test_target,
                            values_format='d',
                            display_labels=["Did not leave","Left"])
        return
