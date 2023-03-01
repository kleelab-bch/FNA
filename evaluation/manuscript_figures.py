import os
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

font = {'size':'11', 'weight':'normal',}
matplotlib.rc('font', **font)


def perform_wilcoxon_signed_rank_sum_test():
    a_list = [0.687, 0.2, 0.634, 0.367, 0.364, 0.667, 0.623]
    b_list = [0.365, 0.202, 0.464, 0.356, 0.212, 0.407, 0.481]
