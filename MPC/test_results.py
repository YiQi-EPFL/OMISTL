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


def test_strategy(N,n_obs,obs_default=True):
    N = N
    paraset(N=N,n_obs=n_obs,Qs=1,Rs=0,num_probs=20000,obs_default=obs_default)
    #pass the value from config to dict and para
    relative_path = os.getcwd()
    dataset_name = 'MPC_horizon_{}'.format(N)
    # dataset_name = 'NSTL_{}_horizon_{}'.format(Qs, N)
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
    Ak = param_dict['Ak']
    Bk = param_dict['Bk']
    Q = param_dict['Q']
    R = param_dict['R']
    n_obs = param_dict['n_obs']
    umin = param_dict['umin']
    umax = param_dict['umax']
    sampled_params = config[2]
    n_obs = config[3]
    num_probs = config[4]
    border_size = config[5]
    box_buffer = config[6]
    min_box_size = config[7]
    max_box_size = config[8]
    posmin = config[9]
    posmax = config[10]
    velmin = config[11]
    velmax = config[12]
    n = config[13]
    m= config[14]

    obs_fix = config[15]
    xg_fix = config[16]
    if obs_fix:
        obstacles = config[-1]

    config_fn = os.path.join(relative_path, 'config', dataset_name+'.p')#

    prob = MPC(config=config_fn)
    #create numpy containers for data: (params, x, u, y, J*, solve_time)
    params = {}
    if 'x0' in sampled_params:
        params['x0'] = np.zeros((num_probs,2*n))
    if 'xg' in sampled_params:
        params['xg'] = np.zeros((num_probs,2*n))
    if 'obstacles' in sampled_params:
        params['obstacles'] = np.zeros((num_probs, 4, n_obs))

    X = np.zeros((num_probs, 2*n, N));
    U = np.zeros((num_probs, m, N-1))
    Y = np.zeros((num_probs, 4*n_obs, N-1)).astype(int)
    Z = np.zeros((num_probs, 2*n_obs, N-1)).astype(int)

    costs = np.zeros(num_probs)
    solve_times = np.zeros(num_probs)

    prob.sampled_params = ['x0', 'xg', 'obstacles']

    #solving MICP
    ii_toggle = 0
    obs_new_ct = 5
    ii=0
    obstacles = config[-1]

    if obs_fix:
        for ii in tqdm(range(num_probs)):
            x0 = findIC(obstacles, posmin, posmax, velmin, velmax)
            params['obstacles'][ii,:] = np.reshape(np.concatenate(obstacles, axis=0), (n_obs,4)).T
            p_dict = {}
            params['x0'][ii,:] = x0
            xg= findIC(obstacles, posmin, posmax, velmin, velmax)
            params['xg'][ii,:] = xg

            p_dict['x0'] = params['x0'][ii,:]
            p_dict['xg'] = params['xg'][ii,:]
            p_dict['obstacles'] = params['obstacles'][ii,:]

            prob_success = False
            try:
                # with time_limit(20):
                prob_success, cost, solve_time, optvals = prob.solve_stl(p_dict, solver=cp.GUROBI)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print('solver failed at '.format(ii))

            if prob_success:
                costs[ii] = cost; solve_times[ii] = solve_time
                X[ii,:,:], U[ii,:,:], Y[ii,:,:], Z[ii,:,:] = optvals
                ii += 1
    else:
        print('choose to fix obstalce')

    ## shuffle the data because of the spatial orders
    num_train = int(num_probs*0.9)
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

    train_params = {}; test_params = {}
    if 'x0' in sampled_params:
        train_params['x0'] = params['x0'][:num_train,:]
        test_params['x0'] = params['x0'][num_train:,:]
    if 'obstacles' in sampled_params:
        train_params['obstacles'] = params['obstacles'][:num_train,:]
        test_params['obstacles'] = params['obstacles'][num_train:,:]
    if 'xg' in sampled_params:
        train_params['xg'] = params['xg'][:num_train,:]
        test_params['xg'] = params['xg'][num_train:,:]

    train_data = [train_params]
    train_data += [X[:num_train,:,:], U[:num_train,:,:], Y[:num_train,:,:],Z[:num_train,:,:]]
    train_data += [costs[:num_train], solve_times[:num_train]]

    test_data = [test_params]
    test_data += [X[num_train:,:,:], U[num_train:,:,:], Y[num_train:,:,:], Z[:num_train,:,:]]
    test_data += [costs[num_train:], solve_times[num_train:]]

    train_file = open(train_fn,'wb')
    pickle.dump(train_data,train_file); train_file.close()

    test_file = open(test_fn, 'wb')
    pickle.dump(test_data,test_file); test_file.close()

    relative_path = os.getcwd()
    dataset_name = 'MPC_horizon_{}'.format(N)
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
    # prob_features = ['x0','obstacles_map']

    MPC_obj = OMISTL(system, prob, prob_features)

    n_features = 33
    MPC_obj.construct_strategies(n_features, train_data)
    print(MPC_obj.n_strategies)
    MPC_obj.setup_network()

    MPC_obj.training_params['TRAINING_ITERATIONS'] = 300
    MPC_obj.train(train_data=train_data, verbose=True)
    outfile = open(config_fn, "rb")
    config = pickle.load(outfile)
    velmin = -0.2
    velmax = 0.2
    posmin = np.zeros(2)
    n_obs = config[1][5]

    ft2m = 0.3048
    posmax = ft2m * np.array([12., 9.])
    max_box_size = 0.75
    min_box_size = 0.25
    box_buffer = 0.025
    border_size = 0.05

    obstacles = config[-1]
    n_test = 200
    framework = 'OMISTL'
    n_succ = 0
    count = 0
    gurobi_fail = 0

    costs = []
    total_time_ML = []
    num_solves = []

    cost_ratios = []
    costs_ip = []
    total_time_ip = []

    for ii in tqdm(range(n_test)):
        if ii % 1000 == 0:
            print('{} / {}'.format(ii, n_test))
        prob_params = {}
        for k in p_test.keys():
            prob_params[k] = p_test[k][ii]

        try:
            prob_success, cost, total_time, n_evals, optvals, y_guess = MPC_obj.forward(prob_params, max_evals=10,
                                                                                        solver=cp.MOSEK)
            if prob_success:
                n_succ += 1
                costs += [cost]
                total_time_ML += [total_time]
                num_solves += [n_evals]

                true_cost = cost_test[ii]
                costs_ip += [true_cost]
                total_time_ip += [times_test[ii]]

                cost_ratios += [cost / true_cost]
            count += 1
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('Solver failed at {}'.format(ii))
            gurobi_fail += 1
            continue

    costs = np.array(costs)
    cost_ratios = np.array(cost_ratios)
    total_time_ML = np.array(total_time_ML)
    num_solves = np.array(num_solves, dtype=int)

    costs_ip = np.array(costs_ip)
    total_time_ip = np.array(total_time_ip)

    percentage = 100 * float(n_succ) / float(count)
    dict = {'framework': framework, 'costs': costs, 'total_time_ML': total_time_ML, 'num_solves': num_solves,
            'costs_ip': costs_ip, 'total_time_ip': total_time_ip, 'cost_ratios': cost_ratios, 'percentage': percentage}
    f_save = open(dataset_fn + '/result.pkl', 'wb')
    pickle.dump(dict, f_save)
    f_save.close()
