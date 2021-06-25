import pandas as pd
import numpy as np
import sklearn
import torch
import random
import argparse
import json
import os
import sys
import time
from Utils.logger import set_logging_defaults
import logging

def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="python run.py mode")
    
    parser.add_argument(
        'mode', type=str,choices=['gen_data','train_crime','train_priority','train_mixed','eval','record'])   

def main(args):
    flags = parse_args(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda and flags.device == 'gpu' else "cpu")
    configs=vars(flags)
    ## data save path ##
    savepath=os.path.join(os.path.dirname(os.path.abspath(__file__)),'training_data')
    if os.path.exists(savepath) == False:
        os.mkdir(savepath)
    ####################

    ## logger ##
    set_logging_defaults(savepath)
    logger = logging.getLogger('main')
    ############


if __name__ == '__main__':
    main(sys.argv[1:])

