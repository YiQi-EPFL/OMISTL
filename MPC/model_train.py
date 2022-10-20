from para import paraset
import os
import pickle
import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import numpy as np
from solvers.OMISTL import OMISTL

import cvxpy as cp
from utils import *
import pickle, os
from MPC_prob import MPC

from tqdm import tqdm
from para import paraset
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'#不设这个解不出#


def train(N,n_obs):
    N = N
    paraset(N=N, n_obs=n_obs,Qs=1,Rs=0,num_probs=20000,obs_default=False)
    #pass the value from config to dict and para
    relative_path = os.getcwd()
    dataset_name = 'MPC_horizon_{}_obs_{}'.format(N,n_obs)
    config_fn = os.path.join(relative_path, 'config', dataset_name+'.p')

    # config = [dataset_name, [prob_params] ,sampled_params]
    outfile = open(config_fn,"rb")
    config= pickle.load(outfile)
    outfile.close()

    train_fn = 'train_horizon_{}.p'.format(N)
    train_fn = os.path.join(relative_path, 'data', dataset_name, train_fn)
    test_fn = 'test_horizon_{}.p'.format(N)
    test_fn = os.path.join(relative_path, 'data', dataset_name, test_fn)

    all_params = ['N', 'Ak', 'Bk', 'Q', 'R', 'n_obs', \
                'posmin', 'posmax', 'velmin', 'velmax', \
                'umin', 'umax']

    param_dict={}
    i=0
    len_para = len(all_params)
    for param in all_params:
        param_dict[param]= config[1][i]
        i+=1
        if i == len_para:
            break

    N = param_dict['N']
    obs_fix = config[15]
    xg_fix = config[16]
    if obs_fix:
        obstacles = config[-1]

    relative_path = os.getcwd()
    dataset_name = 'MPC_horizon_{}_obs_{}'.format(N,n_obs)
    # dataset_name = 'NSTL_{}_horizon_{}'.format(Qs, N)
    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')
    prob = MPC(config=config_fn)  # use default config, pass different config file oth.

    relative_path = os.getcwd()
    dataset_fn = relative_path + '/data/' + dataset_name

    ##load train data
    train_file = open(dataset_fn + '/train_horizon_{}.p'.format(N), 'rb')
    # train_file = open(dataset_fn+'/train.p','rb')
    train_data = pickle.load(train_file)
    p_train, x_train, u_train, y_train, z_train, cost_train, times_train = train_data
    train_file.close()

    x_train = train_data[1]  # X sequence
    y_train = train_data[3]  # Y sequence

    ##load test data
    test_file = open(dataset_fn + '/test_horizon_{}.p'.format(N), 'rb')
    # test_file = open(dataset_fn+'/test.p','rb')
    # p_test, x_test, u_test, y_test, c_test, times_test = pickle.load(test_file)
    test_data = pickle.load(test_file)
    p_test, x_test, u_test, y_test, z_test, cost_test, times_test = test_data
    test_file.close()

    system = 'MPC'
    prob_features = ['x0', 'xg', 'obstacles']

    MPC_obj = OMISTL(system, prob, prob_features)

    n_features = 2*prob.n*(len(prob_features)-1)+n_obs*4+n_obs
    MPC_obj.construct_strategies(n_features, train_data)
    print('nubmber of strategies:'+ str(MPC_obj.n_strategies))
    print('n_obs: '+ str(n_obs))
    print('N: '+ str(N))
    MPC_obj.setup_network()
    fn_saved = '..\\models\\MPC_horizon_{}_obs_{}.pt'.format(N, n_obs)
    Model_exist = MPC_obj.load_network(fn_saved)
    if Model_exist:
        quit()
    else:
        MPC_obj.training_params['TRAINING_ITERATIONS'] = 300
        MPC_obj.train(train_data=train_data, verbose=True)