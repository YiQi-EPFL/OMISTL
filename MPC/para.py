import numpy as np
import os
import pickle
from utils import *
import numpy as np

def paraset(N,n_obs,Qs,Rs=1000.,num_probs=10000,obs_fix=True,xg_fix=False, obs_default = False):

        d=2
        N = N
        n_obs = n_obs
        n = 2; m = 2
        dh = 0.75
        Ak = np.eye(2*n)
        Ak[:n,n:] = dh*np.eye(n)
        Bk = np.zeros((2*n,m))
        Bk[:n,:] = 0.5*dh**2 * np.eye(n)
        Bk[n:,:] = dh*np.eye(n)

        Q = Qs*np.diag([2,2,1,1.])#cost of the final position #
        R = Rs*np.eye(m)       #control cost

        mass_ff_min = 15.36
        mass_ff_max = 18.08
        mass_ff = 0.5*(mass_ff_min+mass_ff_max)
        thrust_max = 2*1.  # max thrust [N] from two thrusters
        umin = -thrust_max/mass_ff
        umax = thrust_max/mass_ff
        velmin = -0.2
        velmax = 0.2
        posmin = np.zeros(n)

        ft2m = 0.3048
        posmax = ft2m*np.array([12.,9.])
        max_box_size = 0.75
        min_box_size = 0.25
        box_buffer = 0.025
        border_size = 0.05

        prob_params = [N, Ak, Bk, Q, R, n_obs, \
            posmin, posmax, velmin, velmax, umin, umax]

        #setup filenames
        relative_path = os.getcwd()
        if obs_fix:
            if not xg_fix:
                dataset_name = 'MPC_horizon_{}_obs_{}'.format(N,n_obs)

        if not os.path.isdir(os.path.join(relative_path, 'data', dataset_name)):
                os.mkdir(os.path.join(relative_path+'/data/'+dataset_name))

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

        param_dict = {'N':N, 'Ak':Ak, 'Bk':Bk, 'Q':Q, 'R':R, 'n_obs':n_obs, \
            'posmin':posmin, 'posmax':posmax, 'velmin':velmin, 'velmax':velmax, \
            'umin':umin, 'umax':umax}

        #specify which parameters to sample, & their distributions
        sampled_params = ['x0', 'xg', 'obstacles']
        sample_dists = {'x0': lambda: np.hstack((posmin + (posmax-posmin)*np.random.rand(2), \
                       velmin + (velmax-velmin)*np.random.rand(2))) ,\
                       'xg': lambda: np.hstack((0.9*posmax, np.zeros(n))), \
                       'obstacles': lambda: random_obs(n_obs, posmin, posmax, border_size, min_box_size, max_box_size)}

        #specify dataset sizes

        num_probs = num_probs
        num_train = num_probs*0.9; num_test = num_probs*0.1


        #write out solver_config
        config_fn = os.path.join(relative_path, 'config', dataset_name+'.p')

        if obs_fix:
            if obs_default:
                obstacles = \
                [np.array([1.25, 2.00, 1.20, 1.50]),
                 np.array([1.25, 1.75, 0.20, 1.00]),
                 np.array([0.30, 0.80, 1.50, 2.00]),
                 np.array([2.50, 3.25, 1.60, 2.00]),
                 np.array([2.90, 3.25, 2.00, 2.25])
                 ]
                n_obs = len(obstacles)

            else:
                obstacles = random_obs(n_obs, posmin, posmax, border_size, box_buffer, min_box_size, max_box_size)
            config = [dataset_name, prob_params, sampled_params, n_obs, num_probs, border_size, box_buffer, min_box_size, max_box_size, posmin, posmax,
                      velmin, velmax, n, m,
                      obs_fix, xg_fix, obstacles]
        else:
            config = [dataset_name, prob_params, sampled_params, n_obs, num_probs, border_size, box_buffer, min_box_size, max_box_size, posmin, posmax,
                      velmin, velmax, n, m,
                      obs_fix, xg_fix]

        outfile = open(config_fn,"wb")
        pickle.dump(config,outfile); outfile.close()
        return
