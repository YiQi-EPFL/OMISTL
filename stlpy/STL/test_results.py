from para_stl import paraset
import os
import pickle
import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import numpy as np
from solvers.OMISTL import OMISTL
from stl_planner import STL_planner
import cvxpy as cp
import pickle, os

from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'#不设这个解不出#


def test_strategy(spec_name,spec,N, test_number = 0.02):
    N = N
    dataset_name = '{}_horizon_{}'.format(spec_name, N)

    # 加载设置文件地址，config = [dataset_name, prob_params, sampled_params]#
    relative_path = os.getcwd()
    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')

    config_file = open(config_fn, 'rb')
    config = pickle.load(config_file)

    dataset_name = config[0]
    config_file.close()  # 读取设置#

    #pass the value from config to dict and para
    relative_path = os.getcwd()
    dataset_name = '{}_horizon_{}'.format(spec_name,N)
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
    if obs_fix:
        obstacles = config[-1]

    relative_path = os.getcwd()
    # dataset_name = 'NSTL_{}_horizon_{}'.format(Qs, N)
    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')
    prob = STL_planner(config=config_fn, spec=spec)
    prob.init_stl_problem()

    relative_path = os.getcwd()
    dataset_fn = relative_path + '/data/' + dataset_name

    ##load train data
    train_file = open(dataset_fn + '/train_horizon_{}.p'.format(N), 'rb')
    # train_file = open(dataset_fn+'/train.p','rb')
    train_data = pickle.load(train_file)
    p_train, x_train, u_train, z_train, cost_train, times_train,vector_train, integer_train = train_data
    train_file.close()


    ##load test data
    test_file = open(dataset_fn + '/test_horizon_{}.p'.format(N), 'rb')
    # test_file = open(dataset_fn+'/test.p','rb')
    # p_test, x_test, u_test, y_test, c_test, times_test = pickle.load(test_file)
    test_data = pickle.load(test_file)
    p_test, x_test, u_test, z_test, cost_test, times_test, vector_test, integer_test = test_data
    test_file.close()

    system = spec_name
    prob_features = ['x0']

    stl_obj = OMISTL(system, prob, prob_features)


    stl_obj.construct_stl_strategies(train_data,4)
    stl_obj.setup_network()
    fn_saved = '..\\..\\models\\{}_horizon_{}.pt'.format(spec_name, N)
    Model_exist = stl_obj.load_network(fn_saved)
    if not Model_exist:
        quit()
    outfile = open(config_fn, "rb")
    config = pickle.load(outfile)
    n_obs = config[1][5]

    ft2m = 0.3048
    num_probs = config[4]
    n_test =int(num_probs * test_number)
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
            prob_success, cost, total_time, solution, optvals_ML = stl_obj.Predict(prob_params,solver=cp.GUROBI,max_evals=16,verbose=False)

            if prob_success:
                n_succ += 1
                costs += [cost]
                total_time_ML += [total_time]


                true_cost = cost_test[ii]
                costs_ip += [true_cost]
                total_time_ip += [times_test[ii]]

                cost_ratios += [cost / true_cost]
            count += 1
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            # print('Solver failed at {}'.format(ii))
            gurobi_fail += 1
            continue

    costs = np.array(costs)
    cost_ratios = np.array(cost_ratios)
    total_time_ML = np.array(total_time_ML)
    num_solves = np.array(num_solves, dtype=int)

    costs_ip = np.array(costs_ip)
    total_time_ip = np.array(total_time_ip)

    percentage = 100 * float(n_succ) / float(count)
    dict = {'framework': framework, 'N': N, 'n_obs': n_obs, 'costs': costs, 'total_time_ML': total_time_ML,
            'num_solves': num_solves, 'costs_ip': costs_ip, 'total_time_ip': total_time_ip, 'cost_ratios': cost_ratios,
            'strategies': stl_obj.n_strategies, 'percentage': percentage, }
    f_save = open(dataset_fn + '/result.pkl', 'wb')
    pickle.dump(dict, f_save)
    f_save.close()

def load_result(spec_name,N):
    N=N
    spec_name = spec_name
    relative_path = os.getcwd()
    dataset_name = '{}_horizon_{}'.format(spec_name, N)
    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')
    relative_path = os.getcwd()
    dataset_fn = relative_path + '/data/' + dataset_name
    f_read = open(dataset_fn + '/result.pkl', 'rb')
    results = pickle.load(f_read)
    f_read.close()
    return results