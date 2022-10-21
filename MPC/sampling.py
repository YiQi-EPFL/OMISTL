import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import numpy as np

import sys
sys.path.append("..")
from solvers.OMISTL import OMISTL
import cvxpy as cp
from utils import *
import pickle, os
from MPC_prob import MPC

from tqdm import tqdm
from para import paraset
from test_results import load_result
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'#不设这个解不出#


def sampling(N,n_obs):
    N= N
    n_obs = n_obs
    paraset(N=N, n_obs=n_obs,Qs=1,Rs=0,num_probs=20000,obs_default=False)
    # pass the value from config to dict and para
    relative_path = os.getcwd()
    dataset_name = 'MPC_horizon_{}_obs_{}'.format(N, n_obs)
    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')

    outfile = open(config_fn, "rb")
    config = pickle.load(outfile)
    outfile.close()

    train_fn = 'train_horizon_{}.p'.format(N)
    train_fn = os.path.join(relative_path, 'data', dataset_name, train_fn)
    test_fn = 'test_horizon_{}.p'.format(N)
    test_fn = os.path.join(relative_path, 'data', dataset_name, test_fn)

    all_params = ['N', 'Ak', 'Bk', 'Q', 'R', 'n_obs', \
                  'posmin', 'posmax', 'velmin', 'velmax', \
                  'umin', 'umax']

    param_dict = {}
    i = 0
    len_para = len(all_params)
    for param in all_params:
        param_dict[param] = config[1][i]
        i += 1
        if i == len_para:
            break

    N = param_dict['N']
    sampled_params = config[2]
    n_obs = config[3]
    num_probs = config[4]
    posmin = config[9]
    posmax = config[10]
    velmin = config[11]
    velmax = config[12]
    n = config[13]
    m = config[14]

    obs_fix = config[15]
    xg_fix = config[16]
    if obs_fix:
        obstacles = config[-1]

    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')  #

    prob = MPC(config=config_fn)
    # create numpy containers for data: (params, x, u, y, J*, solve_time)
    params = {}
    if 'x0' in sampled_params:
        params['x0'] = np.zeros((num_probs, 2 * n))
    if 'xg' in sampled_params:
        params['xg'] = np.zeros((num_probs, 2 * n))
    if 'obstacles' in sampled_params:
        params['obstacles'] = np.zeros((num_probs, 4, n_obs))

    X = np.zeros((num_probs, 2 * n, N));
    U = np.zeros((num_probs, m, N - 1))
    Y = np.zeros((num_probs, 4 * n_obs, N - 1)).astype(int)
    Z = np.zeros((num_probs, 2 * n_obs, N - 1)).astype(int)

    costs = np.zeros(num_probs)
    solve_times = np.zeros(num_probs)

    prob.sampled_params = ['x0', 'xg', 'obstacles']

    # solving MICP
    obstacles = config[-1]

    if obs_fix:
        for ii in tqdm(range(num_probs)):
            x0 = findIC(obstacles, posmin, posmax, velmin, velmax)
            params['obstacles'][ii, :] = np.reshape(np.concatenate(obstacles, axis=0), (n_obs, 4)).T
            p_dict = {}
            params['x0'][ii, :] = x0
            xg = findIC(obstacles, posmin, posmax, velmin, velmax)
            params['xg'][ii, :] = xg

            p_dict['x0'] = params['x0'][ii, :]
            p_dict['xg'] = params['xg'][ii, :]
            p_dict['obstacles'] = params['obstacles'][ii, :]

            prob_success = False
            try:
                # with time_limit(20):
                prob_success, cost, solve_time, optvals = prob.solve_stl(p_dict, solver=cp.GUROBI)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print('solver failed at '.format(ii))

            if prob_success:
                costs[ii] = cost;
                solve_times[ii] = solve_time
                X[ii, :, :], U[ii, :, :], Y[ii, :, :], Z[ii, :, :] = optvals
                ii += 1
    else:
        print('choose to fix obstalce')

    ## shuffle the data because of the spatial orders
    num_train = int(num_probs * 0.9)
    arr = np.arange(num_probs)
    np.random.shuffle(arr)

    if 'x0' in sampled_params:
        params['x0'] = params['x0'][arr]
    if 'xg' in sampled_params:
        params['xg'] = params['xg'][arr]
    if 'obstacles' in sampled_params:
        params['obstacles'] = params['obstacles'][arr]

    costs = costs[arr]
    solve_times = solve_times[arr]

    X = X[arr]
    U = U[arr]
    Y = Y[arr]
    Z = Z[arr]

    train_params = {};
    test_params = {}
    if 'x0' in sampled_params:
        train_params['x0'] = params['x0'][:num_train, :]
        test_params['x0'] = params['x0'][num_train:, :]
    if 'obstacles' in sampled_params:
        train_params['obstacles'] = params['obstacles'][:num_train, :]
        test_params['obstacles'] = params['obstacles'][num_train:, :]
    if 'xg' in sampled_params:
        train_params['xg'] = params['xg'][:num_train, :]
        test_params['xg'] = params['xg'][num_train:, :]

    train_data = [train_params]
    train_data += [X[:num_train, :, :], U[:num_train, :, :], Y[:num_train, :, :], Z[:num_train, :, :]]
    train_data += [costs[:num_train], solve_times[:num_train]]

    test_data = [test_params]
    test_data += [X[num_train:, :, :], U[num_train:, :, :], Y[num_train:, :, :], Z[:num_train, :, :]]
    test_data += [costs[num_train:], solve_times[num_train:]]

    train_file = open(train_fn, 'wb')
    pickle.dump(train_data, train_file);
    train_file.close()

    test_file = open(test_fn, 'wb')
    pickle.dump(test_data, test_file);
    test_file.close()
