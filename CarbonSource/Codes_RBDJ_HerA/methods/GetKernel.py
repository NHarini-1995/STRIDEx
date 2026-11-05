
#==========================================
#Function to perform bayesian opitmization and generate experiments
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
from methods.Kernel import MixtureViaSumAndProduct, CategoryOverlapKernel, LatentEmbeddingKernel, LatentEmbeddingKernel_precompute
from methods.SamplingCategorical import compute_prob_dist_and_draw_hts, distr
from methods.InitialData_Gen import initialize
from matplotlib import pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from paramz.caching import Cache_this


def get_kernel(data_param, cont_kernel_name):
    default_cont_lengthscale = [0.5] * data_param['Nx']  # cont lengthscale
    default_variance = 1  # Variance is the multiplication factor that appears in the kernel
    
    if data_param['approach_type'] == 'CoCa':
        continuous_dims = list(range(data_param['Nc'], data_param['nDim']))
        categorical_dims = list(range(data_param['Nc']))

    elif data_param['approach_type'] == 'Co':
        continuous_dims = list(range(0, data_param['Nx']))

    elif data_param['approach_type'] in ['CoCa_ICM', 'CoCa_LCM']:
        continuous_dims = list(range(data_param['Nc'], data_param['nDim']))
        categorical_dims = list(range(data_param['Nc']))
        # TL_dims = list(range(data_param['nDim'], data_param['nDim']+1))
    elif data_param['approach_type'] == 'Co_ICM' or data_param['approach_type'] == 'Co_LCM':
        continuous_dims = list(range(0, data_param['Nx']))
        # TL_dims = list(range(data_param['Nx'], data_param['Nx'] + 1))
    elif data_param['approach_type'] in ['CoCa_LEK', 'CoCa_LEK_precompute']:
        continuous_dims = list(range(data_param['Nc'], data_param['nDim']))
        categorical_dims = list(range(data_param['Nc']))
        lek_dims = list(range(data_param['nDim'], data_param['nDim']+data_param['Noutput']))
    elif data_param['approach_type'] in ['Co_LEK', 'Co_LEK_precompute']:
        continuous_dims = list(range(0, data_param['nDim']))
        lek_dims = list(range(data_param['nDim'], data_param['nDim']+data_param['Noutput']))
    else:
        print("Non Implemented Error")


    mix_value = 0.2
    fix_mix_in_this_iter = False
    if data_param['Nc'] == 1:
        decision = True
    else:
        decision = True

    if cont_kernel_name == 'Matern52':
        k_cont = GPy.kern.Matern52(data_param['Nx'], variance=default_variance, #lengthscale = default_cont_lengthscale,
                                   active_dims=continuous_dims, ARD=True)  # continuous kernel
    elif cont_kernel_name == 'Matern32':
        k_cont = GPy.kern.Matern32(data_param['Nx'],variance=default_variance, #lengthscale = default_cont_lengthscale,
                                   active_dims=continuous_dims, ARD=True)  # continuous kernel
    else:
        k_cont = GPy.kern.RBF(data_param['Nx'], variance=default_variance, #lengthscale = default_cont_lengthscale,
                              active_dims=continuous_dims, ARD=True)  # continuous kernel

    if data_param['approach_type'] == 'CoCa':
        k_cat = CategoryOverlapKernel(data_param['Nc'], active_dims=categorical_dims)  # categorical kernel

        my_kernel_v0 = MixtureViaSumAndProduct(data_param['nDim'], k_cat, k_cont, mix=mix_value,
                                               fix_inner_variances= decision,
                                               fix_mix=fix_mix_in_this_iter)
        white_noise = GPy.kern.White(data_param['nDim'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel = my_kernel_v0 + white_noise
        
    elif data_param['approach_type'] == 'Co':
        white_noise = GPy.kern.White(data_param['nDim'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel = k_cont + white_noise

    elif data_param['approach_type'] == 'CoCa_ICM':
        k_cat = CategoryOverlapKernel(data_param['Nc'], active_dims=categorical_dims)  # categorical kernel
        my_kernel_v0 = MixtureViaSumAndProduct(data_param['nDim'], k_cat, k_cont, mix=mix_value,
                                               fix_inner_variances=decision,
                                               fix_mix=fix_mix_in_this_iter)
        white_noise = GPy.kern.White(data_param['nDim'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel_v1 = my_kernel_v0 + white_noise
        my_kernel =  util.multioutput.ICM(input_dim=data_param['nDim'], num_outputs=data_param['Noutput'],
                                          kernel=my_kernel_v1, W_rank = data_param['W_rank'])
        # Coregion = GPy.kern.Coregionalize(1, data_param['Noutput'], active_dims=[data_param['nDim']])
        # my_kernel =   my_kernel_v1.prod(Coregion) #

    elif data_param['approach_type'] == 'CoCa_LCM':
        k_cat = CategoryOverlapKernel(data_param['Nc'], active_dims=categorical_dims)  # categorical kernel
        my_kernel_v0 = MixtureViaSumAndProduct(data_param['nDim'], k_cat, k_cont, mix=mix_value,
                                               fix_inner_variances=decision,
                                               fix_mix=fix_mix_in_this_iter)
        white_noise = GPy.kern.White(data_param['nDim'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel_v1 = my_kernel_v0 + white_noise
        k_cont2 = GPy.kern.RBF(data_param['Nx'], variance=default_variance, lengthscale = default_cont_lengthscale,
                              active_dims=continuous_dims, ARD=True)
        kernel2 = MixtureViaSumAndProduct(data_param['nDim'], k_cat, k_cont2, mix=mix_value,
                                               fix_inner_variances=False,
                                               fix_mix=fix_mix_in_this_iter)
        my_kernel =  util.multioutput.LCM(input_dim=data_param['nDim'], num_outputs=data_param['Noutput'],
                                          kernels_list= [my_kernel_v1, kernel2], W_rank = data_param['W_rank']) #

    elif data_param['approach_type'] == 'Co_ICM':
        white_noise = GPy.kern.White(data_param['Nx'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        Coregion = GPy.kern.Coregionalize(1, data_param['Noutput'], active_dims=[data_param['nDim']])
        my_kernel_v1 =  k_cont+ white_noise
        my_kernel = my_kernel_v1.prod(Coregion) #(k_cont+ white_noise)**Coregion

    elif data_param['approach_type'] == 'Co_LEK':
        white_noise = GPy.kern.White(data_param['Nx'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel_v1 = k_cont + white_noise
        lek = LatentEmbeddingKernel(data_param['Noutput'], N_latent = data_param['Nlatent'],
                              active_dims= lek_dims)
        my_kernel = my_kernel_v1.prod(lek)

    elif data_param['approach_type'] == 'CoCa_LEK':
        k_cat = CategoryOverlapKernel(data_param['Nc'], active_dims=categorical_dims)  # categorical kernel

        my_kernel_v0 = MixtureViaSumAndProduct(data_param['nDim'], k_cat, k_cont, mix=mix_value,
                                               fix_inner_variances=decision,
                                               fix_mix=fix_mix_in_this_iter)
        white_noise = GPy.kern.White(data_param['nDim'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel_v1 = my_kernel_v0 + white_noise
        if 'Wembed_init' in list(data_param.keys()):
            print("I am here")
            lek = LatentEmbeddingKernel(data_param['Noutput'], N_latent = data_param['Nlatent'],  Wembed_init = data_param['Wembed_init'],
                                        active_dims=lek_dims)
        else:
            lek = LatentEmbeddingKernel(data_param['Noutput'], N_latent=data_param['Nlatent'], active_dims=lek_dims)
        my_kernel = my_kernel_v1.prod(lek)
        
    elif data_param['approach_type'] == 'CoCa_LEK_precompute':
        k_cat = CategoryOverlapKernel(data_param['Nc'], active_dims=categorical_dims)  # categorical kernel

        my_kernel_v0 = MixtureViaSumAndProduct(data_param['nDim'], k_cat, k_cont, mix=mix_value,
                                               fix_inner_variances=decision,
                                               fix_mix=fix_mix_in_this_iter)
        white_noise = GPy.kern.White(data_param['nDim'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel_v1 = my_kernel_v0 + white_noise
        lek = LatentEmbeddingKernel_precompute(data_param['Noutput'], N_latent = data_param['Nlatent'],
                                    N_precompute=data_param['N_precompute'], W_pre=data_param['W_pre'],
                                    LorO=data_param['LorO'], active_dims=lek_dims)
        my_kernel = my_kernel_v1.prod(lek)

    return my_kernel


def Scaling_data(data, result):
    mu_x = np.mean(data, 0)
    std_x = np.std(data, 0)
    mu_y = np.mean(result, 0)
    std_y = np.std(result, 0)
    data_norm = (data - mu_x) / std_x
    result_norm = (result - mu_y) / std_y

    return mu_x, std_x, mu_y, std_y, data_norm, result_norm
