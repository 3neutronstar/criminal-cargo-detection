from Learner.baselearner import BaseLearner

class CrimeLearner(BaseLearner):
    def __init__(self,logger, datapath, savepath, device, configs):
        super(CrimeLearner,self).__init__(logger, datapath, savepath, device, configs)
        
class PriorityLearner(BaseLearner):
    def __init__(self,logger, datapath, savepath, device, configs):
        super(PriorityLearner,self).__init__(logger, datapath, savepath, device, configs)
    
