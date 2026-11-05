
#==========================================
#CoCaBO Algorithm - Acquisition Function Definition
#==========================================

import math

import GPy
import numpy as np
import pandas as pd
import random

from typing import Union, Tuple
from paramz.transformations import Logexp

from methods.UpdateCategoricalWeight import compute_reward_for_all_cat_variable, update_weights_for_all_cat_var
from methods.Kernel import MixtureViaSumAndProduct, CategoryOverlapKernel
from methods.SamplingCategorical import compute_prob_dist_and_draw_hts, distr
from methods.InitialData_Gen import initialize
from methods.AcquisitionFunctions import EI, PI, UCB, AcquisitionOnSubspace
from matplotlib import pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


def ask_tell(data, result, data_param, cont_kernel_name, method, batch_size, Wc_list, gamma_list): #
    #Scaling the data
    mu_x, std_x, mu_y, std_y, data_norm, result_norm = Scaling_data(data, result)
    data_norm[:, 0:data_param['Nc']] = data[:, 0:data_param['Nc']]
    # define kernel

    if data_param['approach_type'] == 'CoCa':
        my_kernel = get_kernel(data_param, cont_kernel_name)
        print(my_kernel)
        # build a GP  model
        gp =  GPy.models.GPRegression(data_norm, result_norm, my_kernel)#GPy.core.gp.GP(data, result, my_kernel, )
        # gp.set_XY(data[:-2, :], result[:-2,])
        gp.optimize(max_iters=10000)
        gp_actual = gp
        print(my_kernel)
        print(gp.kern.MixtureViaSumAndProduct.Mat52.lengthscale)
        count_b = 0
        z_next = np.zeros((batch_size, data_param['nDim']))
        ht_batch_list = []
        ht_list = []
        bounds = data_param['bounds']
        C_list = data_param['C']

        # Compute the probability for each category and Choose categorical variables
        if method == 'thompson_sampling':
            zt, yt = initialize(10000, data_param, seed=random.randint(0, 10000))
            zt[:, 0:data_param['Nc']] = zt[:, 0:data_param['Nc']]

            zt_norm = (zt - mu_x) / std_x
            zt_norm[:, 0:data_param['Nc']] = zt[:, 0:data_param['Nc']]

            y_samp = gp.posterior_samples_f(np.array(zt_norm), batch_size)  # .reshape(2000, 2000)
            zt_thompson_index = np.argmax(np.max(y_samp, axis=1), axis=0)
            zt_thompson = zt[zt_thompson_index, :]
            z_next = zt_thompson
            print(z_next.shape)
            ht_batch_list = ht_batch_list + zt_thompson[:, 0:data_param['Nc']].tolist()
            ht_list = ht_list + zt_thompson[:, 0:data_param['Nc']].astype(int).tolist()
            print(ht_list)
            probabilityDistribution_list = [None] * len(C_list)
            S0 = []
            for j in range(len(C_list)):
                probabilityDistribution_list[j] = distr(Wc_list[j], gamma_list[j])

        elif method == 'constant_liar':
            for bat in range(batch_size):
                curr_h, probabilityDistribution_list, S0 = compute_prob_dist_and_draw_hts(Wc_list, gamma_list, C_list, 1)
                print(curr_h)
                ht_batch_list.append(curr_h)
                # For the selected ht_list get the reward and continuous variable
                # Identify the unique categorical sets sampled and the corresponding count
                # Get continous value for the respective categorical variables sampled

                x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])

                zt, yt = initialize(5000, data_param, seed=random.randint(0, 5000))
                zt[:, 0:data_param['Nc']] = np.array(curr_h)
                # print(zt.shape)
                zt_norm = (zt - mu_x) / std_x
                zt_norm[:, 0:data_param['Nc']] = np.array(curr_h)

                y_samp = gp.posterior_samples_f(np.array(zt_norm), 10000)
                # print(y_samp.shape)
                mu = np.mean(y_samp, 2) #Mean of the 2000 values sampled at every combination of zt
                # print(mu.shape)
                var = np.square(np.std(y_samp, 2))
                # print(var.shape)
                var = np.clip(var, 1e-8, np.inf)
                s = np.sqrt(var)  # type: np.ndarray
                acq_samp = (mu + data_param['trade_off'] * s).flatten() #

                zt_cl_index = np.argmax(acq_samp, axis=0)
                zt_cl = zt[zt_cl_index, :]

                zt_cl_norm = zt_norm[zt_cl_index, :]
                z_next[bat:bat+1, :] = zt_cl
                data_norm = np.concatenate((data_norm, zt_cl_norm.reshape(1,-1)), axis = 0)
                result_norm = np.concatenate((result_norm, mu[zt_cl_index:zt_cl_index+1]), axis = 0)
                gp = GPy.models.GPRegression(data_norm, result_norm,
                                                 my_kernel)  # GPy.core.gp.GP(data, result, my_kernel, )
                gp.optimize()
                print(gp.kern.MixtureViaSumAndProduct.Mat52.lengthscale)

                ht_batch_list_rewards = compute_reward_for_all_cat_variable(curr_h, C_list, data_norm, result_norm, 1)#batch_size

                Wc_list = update_weights_for_all_cat_var(C_list, ht_batch_list_rewards, curr_h, Wc_list, gamma_list,
                                               probabilityDistribution_list, 1, S0=None)#batch_size

        elif method == 'constant_liar_pure':
            for bat in range(batch_size):
                print(bat)

                # curr_h, probabilityDistribution_list, S0 = compute_prob_dist_and_draw_hts(Wc_list, gamma_list, C_list, 1)
                # print(curr_h)
                # curr_h = [curr_h].astype(int)

                # For the selected ht_list get the reward and continuous variable
                # Identify the unique categorical sets sampled and the corresponding count
                # Get continous value for the respective categorical variables sampled

                x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])

                zt, yt = initialize(10000, data_param, seed=random.randint(0, 10000))
                curr_h = zt[:, 0:data_param['Nc']]

                # zt[:, 0:data_param['Nc']] = np.array(curr_h)
                ht_batch_list.append(curr_h.tolist())

                zt_norm = (zt - mu_x) / std_x
                zt_norm[:, 0:data_param['Nc']] = np.array(curr_h)

                y_samp = gp.posterior_samples_f(np.array(zt_norm), 10000)
                # print(y_samp.shape)
                mu = np.mean(y_samp, 2) #Mean of the 2000 values sampled at every combination of zt
                # print(mu.shape)
                var = np.square(np.std(y_samp, 2))
                # print(var.shape)
                var = np.clip(var, 1e-8, np.inf)
                s = np.sqrt(var)  # type: np.ndarray
                acq_samp = (mu + data_param['trade_off'] * s).flatten() #

                zt_cl_index = np.argmax(acq_samp, axis=0)
                zt_cl = zt[zt_cl_index, :]
                print(zt_cl)
                zt_cl_norm = zt_norm[zt_cl_index, :]
                z_next[bat:bat+1, :] = zt_cl
                data_norm = np.concatenate((data_norm, zt_cl_norm.reshape(1,-1)), axis = 0)
                result_norm = np.concatenate((result_norm, mu[zt_cl_index:zt_cl_index+1]-3*s[zt_cl_index:zt_cl_index+1]), axis = 0)
                gp = GPy.models.GPRegression(data_norm, result_norm,
                                                 my_kernel)  # GPy.core.gp.GP(data, result, my_kernel, )

                gp.optimize()

                probabilityDistribution_list = [None]*len(C_list)
                S0 = []
                for j in range(len(C_list)):
                    probabilityDistribution_list[j] = distr(Wc_list[j], gamma_list[j])
                    
        Categorical_dist_param = {'ht_batch_list':ht_batch_list, 'Wc_list': Wc_list,
                                  'ht_list':ht_list,'probabilityDistribution_list':probabilityDistribution_list,'S0':S0}

        return z_next, Categorical_dist_param, gp_actual


    elif data_param['approach_type'] == 'Co':
        my_kernel = get_kernel(data_param, cont_kernel_name)

        # build a GP  model
        gp = GPy.models.GPRegression(data_norm, result_norm, my_kernel)
        gp.set_XY(data_norm, result_norm)
        gp.optimize(max_iters=10000)

        gp_actual = gp
        Yp = gp.predict(data_norm)[0]
        plt.scatter(result_norm, Yp)
        plt.plot(result_norm, result_norm)

        if method == 'thompson_sampling':
            zt, yt = initialize(10000, data_param, seed=random.randint(0, 10000))
            x_sc = (zt - mu_x) / std_x

            y_samp = gp.posterior_samples_f(np.array(x_sc), batch_size)
            # print(y_samp.shape)

            zt_thompson_index = np.argmax(y_samp, axis=0)

            zt_thompson = zt[zt_thompson_index, :]
            # print(zt_thompson)

            z_next = zt_thompson[0,:,:]
            data_norm_ts = (z_next - mu_x)/std_x
            y_next = gp.predict(data_norm_ts[data_param['initN']:data_param['initN']+batch_size,:])[0]

            temp2 = np.concatenate((result_norm,y_next), axis = 0)

        elif method == 'constant_liar':
            def optimiser_func(x, gp, mu_x, std_x):
                x_sc = (x - mu_x)/std_x
                acq = UCB(gp, data_param['trade_off'])
                acq_val = -acq.evaluate(np.atleast_2d(x_sc))
                return acq_val

            bounds = data_param['bounds']
            x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])
            lower_bound = np.asarray(x_bounds)[:, 0].reshape(1, len(x_bounds))
            upper_bound = np.asarray(x_bounds)[:, 1].reshape(1, len(x_bounds))

            z_next = np.zeros((batch_size, data_param['Nx']))
            min_y = np.zeros((batch_size, 1))

            for b in range(batch_size):
                min_val = 1
                min_x = None
                n_restarts = 50
                for x0 in np.random.uniform(lower_bound[:, 0], upper_bound[:, 0], size=(n_restarts, data_param['Nx'])):
                    if data_param['prob_type'] == 'UnConstrained':
                        res = minimize(optimiser_func, x0=x0, args=(gp, mu_x, std_x), method='trust-constr', bounds=x_bounds
                                       , options={'verbose': 1})  #
                    elif data_param['prob_type'] == 'Constrained':
                        res = minimize(optimiser_func, x0=x0, args=(gp, mu_x, std_x), method='trust-constr', bounds=x_bounds,
                                       constraints=data_param['Constrains'], options={'verbose': 1})  #
                    if res.fun < min_val:
                        min_val = res.fun
                        min_x = res.x

                z_next[b, :] = min_x.reshape(1, -1)

                temp = np.concatenate((data, z_next[0:b,:]), axis = 0)
                data_norm_ts = (temp - mu_x)/std_x

                # print(gp.predict(data_norm_ts)[0])
                min_y[b,0] = mu_y #gp.predict(data_norm_ts[data_param['initN']+b-1:data_param['initN']+b,:])[0]
                temp2 =  np.concatenate((result_norm, min_y[0:b,:]), axis = 0)

                gp = GPy.models.GPRegression(data_norm_ts, temp2, my_kernel)
                gp.optimize(max_iters=1000)
                gp_mod = gp

        return z_next, temp2, gp_actual

def get_kernel(data_param, cont_kernel_name):
    default_cont_lengthscale = [0.1] * data_param['Nx']  # cont lengthscale
    default_variance = 1  # Variance is the multiplication factor that appears in the kernel
    if data_param['approach_type'] == 'CoCa':
        continuous_dims = list(range(data_param['Nc'], data_param['nDim']))
        categorical_dims = list(range(data_param['Nc']))
    else:
        continuous_dims = list(range(0, data_param['Nx']))

    print('check2')
    mix_value = 0.5
    fix_mix_in_this_iter = False

    if cont_kernel_name == 'Matern52':
        k_cont = GPy.kern.Matern52(data_param['Nx'], variance=default_variance, lengthscale = default_cont_lengthscale,
                                   active_dims=continuous_dims, ARD=True)  # continuous kernel
    elif cont_kernel_name == 'Matern32':
        k_cont = GPy.kern.Matern32(data_param['Nx'],variance=default_variance,
                                   active_dims=continuous_dims, ARD=True)  # continuous kernel
    else:
        k_cont = GPy.kern.RBF(data_param['Nx'], variance=default_variance,
                              active_dims=continuous_dims, ARD=True)  # continuous kernel

    if data_param['approach_type'] == 'CoCa':
        k_cat = CategoryOverlapKernel(data_param['Nc'], active_dims=categorical_dims)  # categorical kernel
        print(k_cont.lengthscale)
        my_kernel_v0 = MixtureViaSumAndProduct(data_param['nDim'], k_cat, k_cont, mix=mix_value,
                                               fix_inner_variances=True,
                                               fix_mix=fix_mix_in_this_iter)
        white_noise = GPy.kern.White(data_param['nDim'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel = my_kernel_v0 + white_noise
    else:
        white_noise = GPy.kern.White(data_param['nDim'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel = k_cont + white_noise

    return my_kernel


def Scaling_data(data, result):
    mu_x = np.mean(data, 0)
    std_x = np.std(data, 0)
    mu_y = np.mean(result, 0)
    std_y = np.std(result, 0)
    data_norm = (data - mu_x) / std_x
    result_norm = (result - mu_y) / std_y

    return mu_x, std_x, mu_y, std_y, data_norm, result_norm

