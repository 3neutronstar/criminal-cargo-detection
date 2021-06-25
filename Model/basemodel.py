import torch
import torch.nn as nn
from Model.indmodel import *
from Model.relmodel import *
MODEL={
    'crime':CrimeModel,
    'priority':PriorityModel,
    'mixed':MixedModel,
}
