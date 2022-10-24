from MPC.utils import findIC
import numpy as np
from stl_planner import STL_planner
import cvxpy as cp
import os
import pickle



def sampling_stl(spec_name,spec,N):

    dataset_name = '{}_horizon_{}'.format(spec_name,N)

    #加载设置文件地址，config = [dataset_name, prob_params, sampled_params]#
    relative_path = os.getcwd()
    config_fn = os.path.join(relative_path, 'config', dataset_name+'.p')

    config_file = open(config_fn,'rb')
    config = pickle.load(config_file)

    dataset_name =  config[0]
    config_file.close()#读取设置#
    dataset_name, prob_params, sampled_params, n_obs, num_probs, border_size, box_buffer, min_box_size, max_box_size, \
    posmin, posmax, velmin, velmax, n, m, [obstacles,targets] = config
    params = {}
    if 'x0' in sampled_params:
        params['x0'] = np.zeros((num_probs, 4))

    rec = []
    for j in range(len(obstacles)):
        rec.append(obstacles[j])
    if type(targets) is np.ndarray:
        for i in range(len(targets)):
            for j in range(len(targets[i])):
                rec.append(targets[i][j])

    X = np.zeros((num_probs, 4, N))
    U = np.zeros((num_probs, 2, N))
    Y = np.zeros((num_probs, 6, N)).astype(int)

    costs = np.zeros(num_probs)
    solve_times = np.zeros(num_probs)

    integer_data = []
    for i in range(num_probs):
        integer_data.append(0)

    vector_data = []
    for i in range(num_probs):
        vector_data.append(0)

    from tqdm import tqdm

    if 'x0' in sampled_params:
        params['x0'] = np.zeros((num_probs, 4))
    # if 'xg' in sampled_params:
    #     params['xg'] = np.zeros((num_probs,2*n))
    # if 'obstacles' in sampled_params:
    #     params['obstacles'] = np.zeros((num_probs, 4, n_obs))
    prob = STL_planner(config=config_fn, spec=spec)
    solver_fail = 0
    for ii in tqdm(range(num_probs)):
        x0 = findIC(rec, posmin[:2], posmax[:2], velmin, velmax)
        x0[2:] = 0
        # params['obstacles'][ii,:] = np.reshape(np.concatenate(obstacles, axis=0), (n_obs,4)).T
        p_dict = {}
        params['x0'][ii, :] = x0
        # xg= findIC(obstacles, posmin, posmax, velmin, velmax)
        # params['xg'][ii,:] = xg

        p_dict['x0'] = params['x0'][ii, :]
        # p_dict['xg'] = params['xg'][ii,:]
        # p_dict['obstacles'] = params['obstacles'][ii,:]

        prob.init_stl_problem()
        prob_success = False
        try:
            # with time_limit(20):
            prob_success, cost, solve_time, integer, vector, optvals = prob.solve_stl(p_dict, solver=cp.GUROBI,
                                                                                      verbose=False)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            solver_fail += 1
            print('solver failed at '.format(ii))

        if prob_success:
            vector_data[ii] = vector
            integer_data[ii] = integer
            costs[ii] = cost
            solve_times[ii] = solve_time
            X[ii, :, :], U[ii, :, :], Y[ii, :, :] = optvals
            ii += 1

    num_train = int(num_probs * 0.9)
    arr = np.arange(num_probs)
    np.random.shuffle(arr)

    if 'x0' in sampled_params:
        params['x0'] = params['x0'][arr]

    costs = costs[arr]
    solve_times = solve_times[arr]

    X = X[arr]
    U = U[arr]
    Y = Y[arr]

    integer_data = np.array(integer_data)
    integer_data = integer_data[arr]
    vector_data = np.array(vector_data)
    vector_data = vector_data[arr]

    train_params = {}
    test_params = {}
    if 'x0' in sampled_params:
        train_params['x0'] = params['x0'][:num_train, :]
        test_params['x0'] = params['x0'][num_train:, :]

    train_data = [train_params]
    train_data += [X[:num_train, :, :], U[:num_train, :, :], Y[:num_train, :, :]]
    train_data += [costs[:num_train], solve_times[:num_train]]
    train_data += [vector_data[:num_train], integer_data[:num_train]]

    test_data = [test_params]
    test_data += [X[num_train:, :, :], U[num_train:, :, :], Y[num_train:, :, :]]
    test_data += [costs[num_train:], solve_times[num_train:]]
    test_data += [vector_data[num_train:], integer_data[num_train:]]

    train_fn = 'train_horizon_{}.p'.format(N)
    train_fn = os.path.join(relative_path, 'data', dataset_name, train_fn)
    test_fn = 'test_horizon_{}.p'.format(N)
    test_fn = os.path.join(relative_path, 'data', dataset_name, test_fn)

    train_file = open(train_fn, 'wb')
    pickle.dump(train_data, train_file);
    train_file.close()

    test_file = open(test_fn, 'wb')
    pickle.dump(test_data, test_file);
    test_file.close()