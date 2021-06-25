import torch
import argparse
import time
import os
import sys
from Utils.logger import set_logging_defaults
from Utils.seed import fix_seed
from Utils.params import save_params
from Learner.indlearner import CrimeLearner, PriorityLearner
import logging

LEARNER={
    'crime':CrimeLearner,
    'priority':PriorityLearner,
}

def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="python run.py mode")
    
    parser.add_argument(
        'mode', type=str,choices=['gen_data','train_crime','train_priority','train_mixed','eval','record'])   
    #TRAIN SECTION
    parser.add_argument(
        '--seed', type=int, default=1,
        help='fix random seed')
    parser.add_argument(
        '--batch_size', type=int, default=128,
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
        '--lr', type=float, default=0.01,
        help='set learning rate')
    parser.add_argument(
        '--momentum', type=float, default=0.9,
        help='set learning rate')
    parser.add_argument(
        '--nesterov', type=bool, default=False,
        help='set optimizer nesterov momentum')
    parser.add_argument(
        '--weight_decay', type=float, default=5e-4,
        help='set optimizer\'s weight decay')
    parser.add_argument(
        '--epochs', type=int, default=200,
        help='run epochs')
    return parser.parse_known_args(args)[0]  

def main(args):
    flags = parse_args(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda and flags.device =='gpu' else "cpu")
    configs=vars(flags)
    configs['device']=device
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
        current_path=os.path.join(current_path,'drive','MyDrive')
    data_path=os.path.join(current_path,'data','custom_contest')
    save_path=os.path.join(current_path,'training_data')
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    ####################

    ## save configuration ##
    save_params(configs,current_path,time_data)
    ########################

    ## logger ##
    set_logging_defaults(time_data,save_path)
    logger = logging.getLogger('main')
    ############

    ## learner ##
    learner=LEARNER[configs['mode'].split('_')[1]](logger, data_path, save_path, device, configs)
    learner.run()
    #############


if __name__ == '__main__':
    main(sys.argv[1:])

