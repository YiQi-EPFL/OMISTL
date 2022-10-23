
import os
import pickle
from stlpy.benchmark.reach_avoid import ReachAvoid
from stlpy.benchmark.random_multitarget import RandomMultitarget
import numpy as np
from stl_planner import STL_planner
import cvxpy as cp
from solvers.OMISTL import OMISTL




def train_horizon_stl(N, spec,spec_name, device_id=0):
    dataset_name = '{}_horizon_{}'.format(spec_name ,N)
    relative_path = os.getcwd()
    config_fn = os.path.join(relative_path, 'config', dataset_name +'.p')
    config_file = open(config_fn ,'rb')
    config =  pickle.load(config_file)
    dataset_name = config[0]; config_file.close()

    train_fn = 'train_horizon_{}.p'.format(N)
    train_fn = os.path.join(relative_path, 'data', dataset_name, train_fn)
    test_fn = 'test_horizon_{}.p'.format(N)
    test_fn = os.path.join(relative_path, 'data', dataset_name, test_fn)

    train_file = open(train_fn ,'rb')
    train_data = pickle.load(train_file); train_file.close()

    test_file = open(test_fn ,'rb')
    test_data = pickle.load(test_file)

    test_file.close()

    system = spec_name
    prob_features = ['x0']

    prob = STL_planner(config=config_fn , spec=spec)
    Network = OMISTL(system, prob, prob_features)

    Network.construct_stl_strategies(train_data, 4)

    print('Number of strategies for horizon {}: {}'.format(N, Network.n_strategies))

    Network.setup_network(device_id=device_id)

    save_path = 'models/{}_horizon_{}.pt'.format(spec_name,N)
    save_path = os.path.join('../../', save_path)
    Network.model_fn = save_path
    Network.training_params['TRAINING_ITERATIONS'] =500
    Network.train(train_data=train_data, verbose=True)