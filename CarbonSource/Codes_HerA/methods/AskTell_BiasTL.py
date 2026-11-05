
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
from methods.DataPrep import dataprep
from methods.UpdateCategoricalWeight import compute_reward_for_all_cat_variable, update_weights_for_all_cat_var
from methods.Kernel import MixtureViaSumAndProduct, CategoryOverlapKernel, LatentEmbeddingKernel
from methods.SamplingCategorical import compute_prob_dist_and_draw_hts, distr
from methods.InitialData_Gen import initialize
from methods.GetKernel import get_kernel, Scaling_data
from matplotlib import pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from paramz.caching import Cache_this


def ask_tell(data, result, data_param, cont_kernel_name, method, batch_size, Wc_list_init, gamma_list):
    #Scaling the data
    mu_x, std_x, mu_y, std_y, data_norm, result_norm = Scaling_data(data, result)
    # print(data_norm)
    data_norm[:, data_param['Var_unsc']] = data[:, data_param['Var_unsc']]

    sample_size = 7500 # For in-silico sampling required for certain approaches
    TL_options_list = ['CoCa_LEK', 'CoCa_ICM','CoCa_LCM', 'Co_LEK', 'C_ICM', 'Co_LCM', 'Co_LEK_precompute', 'CoCa_LEK_precompute']

    if data_param['BiasTL'] == True:
        assert data_param['approach_type'] in TL_options_list
        if data_param['SampleSystem'] == True:
            ID_bounds = [0, data_param['Noutput']]
        else:
            assert 'SystemID' in list(data_param.keys())
            assert data_param['SystemID'] <= data_param['Noutput']-1
            assert type(data_param['SystemID']) == int

    elif data_param['BiasTL'] == False:
        assert data_param['approach_type'] in ['Co', 'CoCa']
    else:
        print("Define BiasTL as True or False")


    if data_param['approach_type'] in ['CoCa', 'CoCa_LEK', 'CoCa_ICM','CoCa_LCM','CoCa_LEK_precompute']:
        my_kernel = get_kernel(data_param, cont_kernel_name)
        # build a GP  model
        gp =  GPy.models.GPRegression(data_norm, result_norm, my_kernel)
        print(gp['mul.latentVariable.Wembedding'])
        gp.optimize(max_iters=10000)
        print(gp['mul.latentVariable.Wembedding'])
        gp_actual = gp
        print(my_kernel)
        # print(gp['mul.latentVariable.Wembedding'])
        if data_param['approach_type'] in ['CoCa_LEK', 'CoCa_LEK_precompute']:
            Wembed_updated = np.array(gp['mul.latentVariable.Wembedding'])
        else:
            Wembed_updated = []

        count_b = 0
        if data_param['approach_type'] == 'CoCa':
            z_next = np.zeros((batch_size, data_param['nDim']))
        elif data_param['approach_type'] in ['CoCa_ICM', 'CoCa_LCM']:
            z_next = np.zeros((batch_size, data_param['nDim'] + 1))
        elif data_param['approach_type'] in ['CoCa_LEK', 'CoCa_LEK_precompute']:
            z_next = np.zeros((batch_size, data_param['nDim'] + data_param['Noutput']))

        ht_batch_list = []
        ht_list = []
        bounds = data_param['bounds']
        C_list = data_param['C']
        Wc_list = np.copy(Wc_list_init)
        # Compute the probability for each category and Choose categorical variables
        if method == 'thompson_sampling':
            count_b = 0
            count_a = 0
            # Compute the probability for each category and Choose categorical variables

            ht_batch_list, probabilityDistribution_list, S0 = compute_prob_dist_and_draw_hts(Wc_list, gamma_list,
                                                                                             C_list, batch_size)
            ht_list = ht_batch_list
            ht_batch_list = ht_batch_list.astype(int)
            # For the selected ht_list get the reward and continuous variable
            # Identify the unique categorical sets sampled and the corresponding count
            h_unique, h_counts = np.unique(ht_batch_list, return_counts=True, axis=0)

            z_batch_list = []
            # Get continous value for the respective categorical variables sampled
            for idx, curr_h in enumerate(h_unique):
                curr_x_batch_size = h_counts[idx]  # No. of conti varaible to be drawn for a given categorical set
                x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])
                if method == 'thompson_sampling':

                    zt, yt = initialize(sample_size, data_param, seed=random.randint(0, sample_size))
                    zt[:, 0:data_param['Nc']] = curr_h

                    if data_param['approach_type'] != 'CoCa':
                        if data_param['SampleSystem'] == True:
                            sys_id = np.random.randint(0, data_param['Noutput'], sample_size).reshape(-1,1)
                        else:
                            sys_id = data_param['SystemID'] * np.ones((sample_size,1), dtype = int)

                    if data_param['approach_type'] != 'CoCa':
                        if data_param['approach_type'] in ['CoCa_LEK', 'CoCa_LEK_precompute']:
                            sys_id_np = np.zeros((sample_size, data_param['Noutput']))
                            sys_id_np[:, sys_id] = 1
                        else:
                            sys_id_np = sys_id

                        zt = np.concatenate((zt, sys_id_np), axis = 1)

                    zt_norm = (zt - mu_x) / std_x
                    zt_norm[:, data_param['Var_unsc']] = zt[:, data_param['Var_unsc']]

                    y_samp = gp.posterior_samples_f(np.array(zt_norm), curr_x_batch_size)
                    print(y_samp.shape)

                    zt_thompson_index = np.argmax(y_samp, axis=0)

                    zt_thompson = zt[zt_thompson_index, :]
                    print(zt_thompson)
                    count_a = count_a + curr_x_batch_size
                    z_next[count_b:count_a, :] = zt_thompson
                    count_b = count_b + curr_x_batch_size

            print(pd.DataFrame(z_next))

            ht_next_list_array = ht_batch_list

            # ht_list_rewards = compute_reward_for_all_cat_variable(ht_next_list_array, C_list,
            #                                                       data_norm, result_norm, batch_size)
            #
            # Wc_list = update_weights_for_all_cat_var(C_list, ht_list_rewards, ht_list, Wc_list, gamma_list,
            #                                         probabilityDistribution_list,
            #                                          batch_size, S0)

        elif method == 'constant_liar':
            data_norm_org = np.copy(data_norm)
            result_norm_org = np.copy(result_norm)
            for bat in range(batch_size):
                print(bat)
                x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])
                zt, yt = initialize(sample_size, data_param, seed=random.randint(0,sample_size))
                print(zt.shape)
                print(data_param['Nc'])


                if data_param['approach_type'] != 'CoCa':
                    if data_param['SampleSystem'] == True:
                        sys_id = np.random.randint(0, data_param['Noutput'], sample_size).reshape(-1, 1)
                    else:
                        sys_id = data_param['SystemID'] * np.ones((sample_size, 1), dtype=int)

                if data_param['approach_type'] != 'CoCa':
                    if data_param['approach_type'] in ['CoCa_LEK', 'CoCa_LEK_precompute']:
                        sys_id_np = np.zeros((sample_size, data_param['Noutput']))
                        sys_id_np[:, sys_id] = 1
                    else:
                        sys_id_np = sys_id

                    zt = np.concatenate((zt, sys_id_np), axis=1)

                zt_norm = (zt - mu_x) / std_x
                zt_norm[:, data_param['Var_unsc']] = zt[:, data_param['Var_unsc']]
                
                y_samp = gp.posterior_samples_f(np.array(zt_norm), 10000)
                mu = np.mean(y_samp, 2) #Mean of the 10000 values sampled at every combination of zt
                var = np.square(np.std(y_samp, 2))
                var = np.clip(var, 1e-8, np.inf)
                s = np.sqrt(var)
                acq_samp = (mu + data_param['trade_off'] * s).flatten() #UCB(gp, zt_norm, data_param['trade_off']) #(mu + data_param['trade_off'] * s).flatten() #

                zt_cl_index = np.argmax(acq_samp, axis=0)
                zt_cl = zt[zt_cl_index, :]

                curr_h =(zt[zt_cl_index, 0:data_param['Nc']]).astype(np.int64)

                ht_batch_list.append(curr_h.tolist()) #

                print(zt_cl)
                zt_cl_norm = zt_norm[zt_cl_index, :]
                z_next[bat:bat+1, :] = zt_cl
                data_norm = np.concatenate((data_norm, zt_cl_norm.reshape(1,-1)), axis = 0)
                min_value = np.min(y_samp, 2)[zt_cl_index].reshape(-1,1)
                result_norm = np.concatenate((result_norm, mu[zt_cl_index:zt_cl_index+1]-3*s[zt_cl_index:zt_cl_index+1]), axis = 0) #
                #mu[zt_cl_index:zt_cl_index+1]-3*s[zt_cl_index:zt_cl_index+1]
                #mu_y.reshape(-1,1)- 3*std_y.reshape(-1,1)
                gp = GPy.models.GPRegression(data_norm, result_norm,my_kernel)  # GPy.core.gp.GP(data, result, my_kernel, )
                gp.optimize()
                # print(curr_h)
                # probabilityDistribution_list = [None]*len(C_list)
                # S0 = []
                # Wc_list = Wc_list_init
                # for j in range(len(C_list)):
                #     probabilityDistribution_list[j] = distr(Wc_list[j], gamma_list[j])

                dummy_h, probabilityDistribution_list, S0 = compute_prob_dist_and_draw_hts(Wc_list_init, gamma_list,C_list, 1)

                ht_batch_list_rewards = compute_reward_for_all_cat_variable(curr_h, C_list, data_norm_org, result_norm_org, 1)  # batch_size

                Wc_list = update_weights_for_all_cat_var(C_list, ht_batch_list_rewards, curr_h, Wc_list, gamma_list,
                                                         probabilityDistribution_list, 1, S0)

                # print(Wc_list)

        elif method == 'MAB_constant_liar':
            data_norm_org = np.copy(data_norm)
            result_norm_org = np.copy(result_norm)
            for bat in range(batch_size):
                curr_h, probabilityDistribution_list, S0 = compute_prob_dist_and_draw_hts(Wc_list, gamma_list, C_list, 1)  # _init

                print(curr_h)
                ht_batch_list.append(curr_h)
                # For the selected ht_list get the reward and continuous variable
                # Identify the unique categorical sets sampled and the corresponding count
                # Get continous value for the respective categorical variables sampled

                x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])

                zt, yt = initialize(sample_size, data_param, seed=random.randint(0, sample_size))
                zt[:, 0:data_param['Nc']] = np.array(curr_h)
                if data_param['approach_type'] != 'CoCa':
                    if data_param['SampleSystem'] == True:
                        sys_id = np.random.randint(0, data_param['Noutput'], sample_size).reshape(-1, 1)
                    else:
                        sys_id = data_param['SystemID'] * np.ones((sample_size, 1), dtype=int)

                if data_param['approach_type'] != 'CoCa':
                    if data_param['approach_type'] in ['CoCa_LEK', 'CoCa_LEK_precompute']:
                        sys_id_np = np.zeros((sample_size, data_param['Noutput']))
                        sys_id_np[:, sys_id] = 1
                    else:
                        sys_id_np = sys_id

                    zt = np.concatenate((zt, sys_id_np), axis=1)
                # print(zt.shape)
                zt_norm = (zt - mu_x) / std_x
                zt_norm[:,data_param['Var_unsc']] = zt[:,data_param['Var_unsc']]

                y_samp = gp.posterior_samples_f(np.array(zt_norm), 10000)
                # print(y_samp.shape)
                mu = np.mean(y_samp, 2)  # Mean of the 10000 values sampled at every combination of zt
                # print(mu.shape)
                var = np.square(np.std(y_samp, 2))
                # print(var.shape)
                var = np.clip(var, 1e-8, np.inf)
                s = np.sqrt(var)  # type: np.ndarray

                acq_samp = (mu + data_param['trade_off'] * s).flatten()  #UCB(gp, zt_norm, data_param['trade_off']) #

                zt_cl_index = np.argmax(acq_samp, axis=0)
                zt_cl = zt[zt_cl_index, :]

                zt_cl_norm = zt_norm[zt_cl_index, :]
                z_next[bat:bat + 1, :] = zt_cl
                data_norm = np.concatenate((data_norm, zt_cl_norm.reshape(1, -1)), axis=0)
                result_norm = np.concatenate((result_norm, mu[zt_cl_index:zt_cl_index + 1]- 3*s[zt_cl_index:zt_cl_index + 1]), axis=0) #3*
                gp = GPy.models.GPRegression(data_norm, result_norm, my_kernel)  # GPy.core.gp.GP(data, result, my_kernel, )
                gp.optimize()

                ht_batch_list_rewards = compute_reward_for_all_cat_variable(curr_h, C_list, data_norm_org, result_norm_org, 1)  # batch_size

                Wc_list = update_weights_for_all_cat_var(C_list, ht_batch_list_rewards, curr_h, Wc_list, gamma_list,
                                                         probabilityDistribution_list, 1, S0)  # batch_size
            print(curr_h)
            print(Wc_list)
        else:
            print("Check parameters error")

        Categorical_dist_param = {'ht_batch_list': ht_batch_list, 'Wc_list': Wc_list_init,
                                  'ht_list': ht_list, 'probabilityDistribution_list': probabilityDistribution_list,
                                  'S0': S0,'Wembed_updated': Wembed_updated}


    elif data_param['approach_type'] in ['Co', 'Co_LEK', 'Co_ICM', 'Co_LCM']:
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
            zt, yt = initialize(sample_size, data_param, seed=random.randint(0, sample_size))
            if data_param['approach_type'] != 'Co':
                if data_param['SampleSystem'] == True:
                    sys_id = np.random.randint(0, data_param['Noutput'], sample_size).reshape(-1, 1)
                else:
                    sys_id = data_param['SystemID'] * np.ones((sample_size, 1))

                if data_param['approach_type'] != 'Co':
                    if data_param['approach_type'] in ['Co_LEK', 'Co_LEK_precompute']:
                        sys_id_np = np.zeros((sample_size, data_param['Noutput']))
                        sys_id_np[:, sys_id] = 1
                    else:
                        sys_id_np = sys_id

                    zt = np.concatenate((zt, sys_id_np), axis=1)

            x_sc = (zt - mu_x) / std_x
            x_sc[:, data_param['Var_unsc']] = zt[:,data_param['Var_unsc']]
            y_samp = gp.posterior_samples_f(np.array(x_sc), batch_size)

            zt_thompson_index = np.argmax(y_samp, axis=0)
            zt_thompson = zt[zt_thompson_index, :]

            z_next = zt_thompson[0,:,:]
            data_norm_ts = (z_next - mu_x)/std_x
            data_norm_ts[:, data_param['Var_unsc']] = z_next[:,data_param['Var_unsc']]
            
            y_next = gp.predict(data_norm_ts[data_param['initN']:data_param['initN']+batch_size,:])[0]

            temp2 = np.concatenate((result_norm,y_next), axis = 0)

        elif method == 'constant_liar':
            def optimiser_func(x, gp, mu_x, std_x, sys_id):
                if sys_id != None:
                    x_sc = np.concatenate((x_sc, sys_id), axis=1)
                x_sc = (x - mu_x) / std_x
                x_sc[:, data_param['Var_unsc']] = x[:, data_param['Var_unsc']]
                
                acq_val = UCB(gp, np.atleast_2d(x_sc), data_param['trade_off'])
                # acq_val = -acq.evaluate(np.atleast_2d(x_sc))
                return -acq_val

            bounds = data_param['bounds']
            x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])
            lower_bound = np.asarray(x_bounds)[:, 0].reshape(1, len(x_bounds))
            upper_bound = np.asarray(x_bounds)[:, 1].reshape(1, len(x_bounds))

            if data_param['approach_type'] != 'Co':
                if data_param['SampleSystem'] == True:
                    sys_id = np.random.randint(0, data_param['Noutput'], sample_size).reshape(-1, 1)
                else:
                    sys_id = data_param['SystemID'] * np.ones((sample_size, 1))
                if data_param['approach_type'] != 'Co':
                    if data_param['approach_type'] in ['Co_LEK', 'Co_LEK_precompute']:
                        sys_id_np = np.zeros((sample_size, data_param['Noutput']))
                        sys_id_np[:, sys_id] = 1
                    else:
                        sys_id_np = sys_id
            else:
                sys_id_np = None

            # z_next = np.zeros((batch_size, data_param['Nx']))
            min_y = np.zeros((batch_size, 1))

            for b in range(batch_size):
                min_val = 1
                min_x = None
                n_restarts = 50
                for x0 in np.random.uniform(lower_bound[:, 0], upper_bound[:, 0], size=(n_restarts, data_param['Nx'])):
                    if data_param['prob_type'] == 'UnConstrained':
                        res = minimize(optimiser_func, x0=x0, args=(gp, mu_x, std_x, sys_id_np), method='trust-constr', bounds=x_bounds
                                       , options={'verbose': 1})  #
                    elif data_param['prob_type'] == 'Constrained':
                        res = minimize(optimiser_func, x0=x0, args=(gp, mu_x, std_x, sys_id_np), method='trust-constr', bounds=x_bounds,
                                       constraints=data_param['Constrains_function'], options={'verbose': 1})  #
                    if res.fun < min_val:
                        min_val = res.fun
                        min_x = res.x

                z_next[b, :] = min_x.reshape(1, -1)

                temp = np.concatenate((data, z_next[0:b,:]), axis = 0)
                data_norm_ts = (temp - mu_x)/std_x
                data_norm_ts[:, data_param['Var_unsc']] = temp[:, data_param['Var_unsc']]

                min_y[b,0] = np.random.randn(b,1) #gp.predict(data_norm_ts[data_param['initN']+b-1:data_param['initN']+b,:])[0]
                temp2 =  np.concatenate((result_norm, min_y[0:b,:]), axis = 0)

                gp = GPy.models.GPRegression(data_norm_ts, temp2, my_kernel)
                gp.optimize(max_iters=10000)
                gp_mod = gp

        else:
            print("Check parameters error")

        Categorical_dist_param = {'Wc_list':[]}
    else:
        print("Check Parameters Error")

    return z_next, Categorical_dist_param, gp_actual



def UCB(gp, zt_norm, trade_off):
    y_samp = gp.posterior_samples_f(np.array(zt_norm), 10000)
    mu = np.mean(y_samp, 2)  # Mean of the 10000 values sampled at every combination of zt
    # print(mu.shape)
    var = np.square(np.std(y_samp, 2))
    # print(var.shape)
    var = np.clip(var, 1e-8, np.inf)
    s = np.sqrt(var)
    acq_samp = (mu + trade_off * s).flatten() #

    return acq_samp