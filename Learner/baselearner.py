
class BaseLearner:
    def __init__(self,logger, datapath, savepath, device, configs):
        self.datapath=datapath # data load
        self.savepath=savepath # save (tensorboard)
        self.device=device # cuda
        self.configs=configs
        self.logger=logger
        self.input_space=self.train_dataloader.dataset[0][0].size()[0]
        if 'crime' in configs['mode']:
            self.output_space=2
        elif 'priority' in configs['mode']:
            self.output_space=3
