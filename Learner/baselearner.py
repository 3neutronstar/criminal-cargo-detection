
class BaseLearner:
    def __init__(self,logger,time_data, data_path, save_path, device, configs):
        self.data_path=data_path # data load
        self.save_path=save_path # save (tensorboard)
        self.device=device # cuda
        self.configs=configs
        self.logger=logger
        self.time_data=time_data
