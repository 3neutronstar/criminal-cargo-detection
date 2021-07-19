import torch
import torch.nn as nn
from Model.indmodel import *
from Model.mixmodel import *
MODEL={
    'crime':CrimeModel,
    'priority':PriorityModel,
    'mixed':MixedModel,
    'eval':MixedModel,
}
