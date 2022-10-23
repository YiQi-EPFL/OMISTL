import numpy as np
import os
import pickle

from MPC.utils import *
import numpy as np


def paraset(N, n_obs, Q, R, xmin, xmax, vmax, umax, spec_name, obstacles=None, targets=None, num_probs=10000):
    d = 2
    I = np.eye(d)
    z = np.zeros((d, d))
    n_obs = n_obs
    n = 2
    m = 2
    Ak = np.block([[I, I],
                  [z, I]])
    Bk = np.block([[z],
                  [I]])

    Ck = np.block([[I, z],
                   [z, I],
                   [z, z]])
    Dk = np.block([[z],
                   [z],
                   [I]])
    Q = Q
    R = R
    # mass_ff_min = 15.36
    # mass_ff_max = 18.08
    # mass_ff = 0.5 * (mass_ff_min + mass_ff_max)
    # thrust_max = 2 * 1.  # max thrust [N] from two thrusters
    # umin = -thrust_max / mass_ff
    # umax = thrust_max / mass_ff
    umin = -umax
    umax = umax
    velmax = vmax
    velmin = -velmax
    posmin = xmin
    posmax =xmax
    max_box_size = 0.75
    min_box_size = 0.25
    box_buffer = 0.025
    border_size = 0.05

    prob_params = [N, Ak, Bk, Ck, Dk, Q, R, n_obs, \
                   posmin, posmax, velmin, velmax, umin, umax]

    # setup filenames
    relative_path = os.getcwd()

    dataset_name = '{}_horizon_{}'.format(spec_name,N)

    if not os.path.isdir(os.path.join(relative_path, 'data', dataset_name)):
        os.mkdir(os.path.join(relative_path + '/data/' + dataset_name))

    if not os.path.isdir(os.path.join(relative_path, 'config')):
        os.mkdir(os.path.join(relative_path, 'config'))
    train_fn = 'train_horizon_{}.p'.format(N)
    train_fn = os.path.join(relative_path, 'data', dataset_name, train_fn)
    test_fn = 'test_horizon_{}.p'.format(N)
    test_fn = os.path.join(relative_path, 'data', dataset_name, test_fn)

    #   define all possible params that can be varied
    all_params = ['N', 'Ak', 'Bk', 'Q', 'R', 'n_obs', \
                  'posmin', 'posmax', 'velmin', 'velmax', \
                  'umin', 'umax', \
                  'x0', 'xg', 'obstacles']

    param_dict = {'N': N, 'Ak': Ak, 'Bk': Bk, 'Q': Q, 'R': R, 'n_obs': n_obs, \
                  'posmin': posmin, 'posmax': posmax, 'velmin': velmin, 'velmax': velmax, \
                  'umin': umin, 'umax': umax}

    # specify which parameters to sample, & their distributions
    sampled_params = ['x0']
    # write out solver_config
    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')

    config = [dataset_name, prob_params, sampled_params, n_obs, num_probs, border_size, box_buffer, min_box_size,
              max_box_size, posmin, posmax,
              velmin, velmax, n, m, [obstacles,targets]]

    outfile = open(config_fn, "wb")
    pickle.dump(config, outfile);
    outfile.close()
    return
