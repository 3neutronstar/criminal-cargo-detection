import os
import json
def load_params(configs,current_path, file_name):
    ''' replay_name from file_name '''
    with open(os.path.join(current_path, 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs


def save_params(configs,current_path, time_data):
    with open(os.path.join(current_path, 'training_data', '{}.json'.format(time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)