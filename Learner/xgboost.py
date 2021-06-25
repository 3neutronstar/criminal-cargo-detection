
from Learner.baselearner import BaseLearner

from Learner.baselearner import BaseLearner
from Utils.calc_score import calc_score
class XGBoostLearner(BaseLearner):
    def __init__(self,logger, datapath, savepath, device, configs):
        super(XGBoostLearner,self).__init__(logger, datapath, savepath, device, configs)
        
    def run(self):

        return
