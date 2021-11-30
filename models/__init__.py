from typing import Union
import os

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import sys
sys.path.append("..")
from utils import l2_distance