
#==========================================
# Preparing data for different models
#==========================================

import math

import GPy
from GPy import util
import numpy as np
import pandas as pd
import random

from typing import Union, Tuple
from paramz.transformations import Logexp

from methods.UpdateCategoricalWeight import compute_reward_for_all_cat_variable, update_weights_for_all_cat_var
from methods.Kernel import MixtureViaSumAndProduct, CategoryOverlapKernel
from methods.SamplingCategorical import compute_prob_dist_and_draw_hts, distr
from methods.InitialData_Gen import initialize
# from methods.AcquisitionFunctions import EI, PI, UCB, AcquisitionOnSubspace
from matplotlib import pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


def dataprep(data_norm, data_param, cue): #
    if cue == 'OH_Base':
        assert len(data_param['C']) == 1
        categorical_OH = np.zeros((data_norm.shape[0], data_param['C'][0])) #Only the carbon source
        for c in range(data_norm.shape[0]):
            categorical_OH[c, int(data_norm[c, 0])] = 1
        data_norm_OH = np.concatenate((categorical_OH[:, :-1], data_norm[:, 1:-1]), axis=1)
        data_mod = data_norm_OH
        assert data_mod.shape[1] == data_param['nDim']

    elif cue in ['CoCa_Base']:
        data_mod = data_norm[:, :-1]
        assert data_mod.shape[1] == data_param['nDim']

    elif cue == 'OH_Indicator':
        categorical_OH = np.zeros((data_norm.shape[0],data_param['C'][0])) #Noutput
        categorical_OH2 = np.zeros((data_norm.shape[0], data_param['C'][1])) #Carbon sources

        for c in range(data_norm.shape[0]):
            categorical_OH[c, int(data_norm[c, -1])] = 1
            categorical_OH2[c, int(data_norm[c, 0])] = 1
        data_norm_OH = np.concatenate((categorical_OH[:, :-1], categorical_OH2[:, :-1], data_norm[:, 1:-1]), axis=1)
        assert data_norm_OH.shape[1] == data_param['Nx']
        data_mod = data_norm_OH

    elif cue == 'CoCa_Indicator': #First is Noutput, Carbon source list
        data_mod = np.concatenate((data_norm[:, -1:], data_norm[:, :-1]), axis = 1)
        assert data_mod.shape[1] == data_param['nDim']

    elif cue in ['Coreg', 'LCM'] :
        data_mod = data_norm
        assert data_mod.shape[1] == data_param['nDim']+1

    elif cue in ['Co_LEK', 'Co_LEK2'] :
        categorical_OH = np.zeros((data_norm.shape[0], data_param['Noutput'] )) #Carbon source
        categorical_OH2 = np.zeros((data_norm.shape[0], data_param['C'][0])) #

        for c in range(data_norm.shape[0]):
            categorical_OH[c, int(data_norm[c, -1])] = 1
            categorical_OH2[c, int(data_norm[c, 0])] = 1

        data_norm_OH = np.concatenate((categorical_OH2[:, :-1], data_norm[:, 1:-1], categorical_OH), axis=1)

        data_mod = data_norm_OH
        assert data_mod.shape[1] == data_param['nDim'] + data_param['Noutput']

    elif cue in ['CoCa_LEK', 'CoCa_LEK2', 'CoCa_LEK_precompute']:
        data_mod = data_norm
        categorical_OH = np.zeros((data_norm.shape[0], data_param['Noutput']))  # Noutput

        for c in range(data_norm.shape[0]):
            categorical_OH[c, int(data_norm[c, -1])] = 1

        data_norm_OH = np.concatenate((data_norm[:,:-1], categorical_OH), axis=1)
        data_mod = data_norm_OH
        assert data_mod.shape[1] == data_param['nDim']+data_param['Noutput']

    else:
        print("error")

    return data_mod

# def dataprep_ts(X_test, ):
#     if cue == 'OH':
#         categorical_OH = np.zeros((data_norm.shape[0], C_list[0]))
#         categorical_OH2 = np.zeros((data_norm.shape[0], C_list[1]))
#
#         for c in range(data_norm.shape[0]):
#             categorical_OH[c, int(data_norm[c, 0])] = 1
#             categorical_OH2[c, int(data_norm[c, -1])] = 1
#
#     elif cue == 'CoCa_Base':
#
#     elif cue == 'CoCa':
#
#     elif cue == 'Coreg':