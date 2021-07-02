import torch
import argparse
import time
import os
import sys
from Utils.logger import set_logging_defaults
from Utils.seed import fix_seed
from Utils.params import save_params
from Learner.indlearner import CrimeLearner, PriorityLearner
from Learner.mixedlearner import MixedLearner
from Learner.xgboost import XGBoostLearner
from Utils.record import RecordData
from DataProcessing.gen_data import gen_data
import logging

LEARNER={
    'crime':CrimeLearner,
    'priority':PriorityLearner,
    'mixed':MixedLearner,
    'xgboost':XGBoostLearner,
}

def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="python run.py mode")
    
    parser.add_argument(
        'mode', type=str,choices=['gen_data','train_crime','train_priority','train_mixed','eval','record','train_xgboost_crime','train_xgboost_priority'])
    #TRAIN SECTION
    parser.add_argument(
        '--seed', type=int, default=1,
        help='fix random seed')
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='set mini-batch size')
    parser.add_argument(
        '--device', type=str, default='gpu',
        help='choose NeuralNetwork')
    parser.add_argument(
        '--colab', type=bool, default=False,
        help='if you are in colab use it')
    parser.add_argument(
        '--num_workers', type=int, default=3,
        help='number of process you have')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='set learning rate')
    parser.add_argument(
        '--weight_decay', type=float, default=5e-4,
        help='set optimizer\'s weight decay')
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='run epochs')
    parser.add_argument(
        '--lr_decay', type=int, default=5,
        help='run epochs')
    parser.add_argument(
        '--lr_decay_rate', type=float, default=0.9,
        help='run epochs')
    parser.add_argument(
        '--custom_loss', type=str, default='baseline',
        help='run by kd loss or f beta loss')
    parser.add_argument(
        '--preprocess', type=bool, default=False,
        help='using csv(true) or preprocessed data(false)')

    # mode dependancy
    if parser.parse_known_args(args)[0].mode.lower()=='record':
        parser.add_argument(
            '--file_name', type=str, default=None,
            help='read file name')
    elif parser.parse_known_args(args)[0].mode.lower()=='gen_data' or parser.parse_known_args(args)[0].preprocess==True:
        parser.add_argument(
            '--split_ratio', '-sr',type=float, default=0.75,
            help='split ratio for training_set')

    # custom loss dependancy
    if parser.parse_known_args(args)[0].custom_loss.lower()=='kd_loss':
        parser.add_argument(
            '--temperature', type=float, default=20.0,
            help='kd temp')
        parser.add_argument(
            '--alpha', type=float, default=1.5,
            help='kd alpha')
    elif parser.parse_known_args(args)[0].custom_loss.lower()=='fbeta_loss':
        parser.add_argument(
            '--lambda', type=float, default=1.0,
            help='fbeta_loss dependant total loss rate')
        parser.add_argument(
            '--beta', type=float, default=10.0,
            help='f beta score loss')

    
    return parser.parse_known_args(args)[0]  

def main(args):
    flags = parse_args(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda and flags.device =='gpu' else "cpu")
    configs=vars(flags)
    configs['device']=str(device)
    ## seed ##
    fix_seed(seed=flags.seed)
    ##########

    ## time data ##
    time_data = time.strftime(
            '%m-%d_%H-%M-%S', time.localtime(time.time()))
    ###############

    ## data save path ##
    current_path=os.path.dirname(os.path.abspath(__file__))
    if configs['colab']==True:
        par_path=os.path.abspath(os.path.join(current_path, os.pardir))
        current_path=os.path.join(par_path,'drive','MyDrive')
    data_path=os.path.join(current_path,'data','custom_contest')
    save_path=os.path.join(current_path,'training_data')
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    if os.path.exists(os.path.join(save_path,time_data)) == False:
        os.mkdir(os.path.join(save_path,time_data))
    ####################

    ## generate data ##
    if configs['mode']=='gen_data':
        gen_data(data_path,configs)
        exit()
    elif configs['mode']=='record':
        runner=RecordData(data_path,save_path,current_path,configs)
        runner.run()
        exit()
    ###################

    ## save configuration ##
    save_params(configs,current_path,time_data)
    ########################

    ## logger ##
    set_logging_defaults(time_data,save_path)
    logger = logging.getLogger('main')
    ############

    ## learner ##
    learner=LEARNER[configs['mode'].split('_')[1]](logger, time_data, data_path, save_path, device, configs)
    learner.run()
    #############


if __name__ == '__main__':
    main(sys.argv[1:])

