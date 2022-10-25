import os
import cvxpy as cp
import yaml
import pickle
import numpy as np
import sys
from gray import GrayCode
import math
import graycode
import pdb
from sos1 import addsos1constraint

from core import Problem

class MPC(Problem):
    """Class to setup + solve free-flyer problems."""

    def __init__(self, config=None, solver=cp.MOSEK):
        """Constructor for FreeFlyer class.

        Args:
            config: full path to solver_config file. if None, load default solver_config.
            solver: solver object to be used by cvxpy
        """
        super().__init__()

        ## TODO(pculbertson): allow different sets of params to vary.
        if config is None: #use default solver_config
            relative_path = os.path.dirname(os.path.abspath(__file__))
            config = relative_path + '/solver_config/default.p'

        config_file = open(config,"rb")
        self.configs = pickle.load(config_file)
        self.prob_params = self.configs[1]
        self.sampled_params = self.configs[2]
        config_file.close()
        self.init_problem(self.prob_params)

    def init_problem(self,prob_params):
        # setup problem params
        self.n = 2; self.m = 2

        self.N, self.Ak, self.Bk, self.Q, self.R, self.n_obs, \
          self.posmin, self.posmax, self.velmin, self.velmax, \
          self.umin, self.umax = prob_params

        self.H = 64
        self.W = int(self.posmax[0] / self.posmax[1] * self.H)
        self.H, self.W = 32, 32

        self.init_bin_problem()
        self.init_mlopt_problem()
        self.init_stl_problem()

    def init_bin_problem(self):#用来samlple问题，y是variable#
        cons = []

        # Variables
        x = cp.Variable((2*self.n,self.N)) # state
        u = cp.Variable((self.m,self.N-1))  # control
        y = cp.Variable((4*self.n_obs,self.N-1), boolean=True)#solve obstacle bigM var y, turn it into para become mlopt#
        self.bin_prob_variables = {'x':x, 'u':u, 'y':y}

        # Parameters
        x0 = cp.Parameter(2*self.n)
        xg = cp.Parameter(2*self.n)
        obstacles = cp.Parameter((4, self.n_obs))#could add y as para here#
        self.bin_prob_parameters = {'x0': x0, 'xg': xg, 'obstacles': obstacles} #could add y as para here#

        cons += [x[:,0] == x0]

        # Dynamics constraints
        for ii in range(self.N-1):
          cons += [x[:,ii+1] - (self.Ak @ x[:,ii] + self.Bk @ u[:,ii]) == np.zeros(2*self.n)]

        M = 100. # big M value
        for i_obs in range(self.n_obs):
          for i_dim in range(self.n):
            o_min = obstacles[self.n*i_dim,i_obs]
            o_max = obstacles[self.n*i_dim+1,i_obs]

            for i_t in range(self.N-1):
              yvar_min = 4*i_obs + self.n*i_dim
              yvar_max = 4*i_obs + self.n*i_dim + 1

              cons += [x[i_dim,i_t+1] <= o_min + M-M*y[yvar_min,i_t]]
              cons += [-x[i_dim,i_t+1] <= -o_max + M-M*y[yvar_max,i_t]]

          for i_t in range(self.N-1):

            yvar_min, yvar_max = 4*i_obs, 4*(i_obs+1)
            cons += [sum([y[ii,i_t] for ii in range(yvar_min,yvar_max)]) >=1] # micp 问题中整数变量里1的个数#

        # Region bounds
        for kk in range(self.N):
          for jj in range(self.n):
            cons += [self.posmin[jj] - x[jj,kk] <= 0]
            cons += [x[jj,kk] - self.posmax[jj] <= 0]

        # Velocity constraints
        for kk in range(self.N):
          for jj in range(self.n):
            cons += [self.velmin - x[self.n+jj,kk] <= 0]
            cons += [x[self.n+jj,kk] - self.velmax <= 0]

        # Control constraints
        for kk in range(self.N-1):
          cons += [cp.norm(u[:,kk]) <= self.umax]

        lqr_cost = 0.
        # l2-norm of lqr_cost
        for kk in range(self.N):
          lqr_cost += cp.quad_form(x[:,kk]-xg, self.Q)

        for kk in range(self.N-1):
          lqr_cost += cp.quad_form(u[:,kk], self.R)

        self.bin_prob = cp.Problem(cp.Minimize(lqr_cost), cons)

    def init_mlopt_problem(self):#用来预测求解问题，y是para#
        cons = []

        # Variables
        x = cp.Variable((2*self.n,self.N)) # state
        u = cp.Variable((self.m,self.N-1))  # control
        self.mlopt_prob_variables = {'x':x, 'u':u}

        # Parameters
        x0 = cp.Parameter(2*self.n)
        xg = cp.Parameter(2*self.n)
        obstacles = cp.Parameter((4, self.n_obs))
        y = cp.Parameter((4*self.n_obs,self.N-1)) 
        self.mlopt_prob_parameters = {'x0': x0, 'xg': xg,
          'obstacles': obstacles, 'y':y}

        cons += [x[:,0] == x0]

        # Dynamics constraints
        for ii in range(self.N-1):
          cons += [x[:,ii+1] - (self.Ak @ x[:,ii] + self.Bk @ u[:,ii]) == np.zeros(2*self.n)]

        M = 100. # big M value
        for i_obs in range(self.n_obs):
          for i_dim in range(self.n):
            o_min = obstacles[self.n*i_dim,i_obs]
            o_max = obstacles[self.n*i_dim+1,i_obs]

            for i_t in range(self.N-1):
              yvar_min = 4*i_obs + self.n*i_dim
              yvar_max = 4*i_obs + self.n*i_dim + 1

              cons += [x[i_dim,i_t+1] <= o_min + M-M*y[yvar_min,i_t]]
              cons += [-x[i_dim,i_t+1] <= -o_max + M-M*y[yvar_max,i_t]]

          # for i_t in range(self.N-1):
          #   yvar_min, yvar_max = 4*i_obs, 4*(i_obs+1)
          #   cons += [sum([y[ii,i_t] for ii in range(yvar_min,yvar_max)]) == 1]

        # Region bounds
        for kk in range(self.N):
          for jj in range(self.n):
            cons += [self.posmin[jj] - x[jj,kk] <= 0]
            cons += [x[jj,kk] - self.posmax[jj] <= 0]
        
        # Velocity constraints
        for kk in range(self.N):
          for jj in range(self.n):
            cons += [self.velmin - x[self.n+jj,kk] <= 0]
            cons += [x[self.n+jj,kk] - self.velmax <= 0]
            
        # Control constraints
        for kk in range(self.N-1):
          cons += [cp.norm(u[:,kk]) <= self.umax]

        M = 1000. # big M value
        lqr_cost = 0.
        # l2-norm of lqr_cost
        for kk in range(self.N):
          lqr_cost += cp.quad_form(x[:,kk]-xg, self.Q)

        for kk in range(self.N-1):
          lqr_cost += cp.quad_form(u[:,kk], self.R)

        self.mlopt_prob = cp.Problem(cp.Minimize(lqr_cost), cons)

    def init_stl_problem(self):
        cons = []
        # Variables
        num_y = math.ceil(math.log2(self.n * 2))
        x = cp.Variable((2*self.n,self.N)) # state
        u = cp.Variable((self.m,self.N-1))  # control
        # y = cp.Variable((4 * self.n_obs, self.N - 1), boolean=True)
        y = cp.Variable((4 * self.n_obs, self.N-1))

        z = cp.Variable((self.n_obs*num_y, self.N-1), boolean=True) #sos1 encoding#
        self.stl_prob_variables = {'x':x, 'u':u, 'y':y, 'z':z}
        # self.stl_prob_variables = {'x': x, 'u': u, 'y': y}

        # Parameters
        x0 = cp.Parameter(2*self.n)
        xg = cp.Parameter(2*self.n)
        obstacles = cp.Parameter((4, self.n_obs))

        # y = cp.Parameter((4*self.n_obs,self.N-1))

        self.stl_prob_parameters = {'x0': x0, 'xg': xg,
          'obstacles': obstacles}

        cons += [x[:,0] == x0]

        # Dynamics constraints
        for ii in range(self.N-1):
          cons += [x[:,ii+1] - ( self.Ak @ x[:,ii] + self.Bk @ u[:,ii]) == np.zeros(2*self.n)]

        M = 100. # big M value
        for i_obs in range(self.n_obs):
          for i_dim in range(self.n):
            o_min = obstacles[self.n*i_dim,i_obs]
            o_max = obstacles[self.n*i_dim+1,i_obs]

            for i_t in range(self.N-1):
              yvar_min = 4*i_obs + self.n*i_dim
              yvar_max = 4*i_obs + self.n*i_dim + 1

              cons += [x[i_dim,i_t+1] <= o_min + M-M*y[yvar_min,i_t]]
              cons += [-x[i_dim,i_t+1] <= -o_max + M-M*y[yvar_max,i_t]]

        #add sos1 constraints for each timestep
            # for i_t in range(self.N-1):
            #     lambd, sos1_code, sos1_cons = addsos1constraint(self.n * 2)
            #     cons += sos1_cons
            #     cons += [y[4 * i_obs:4 * (i_obs + 1), i_t] == lambd]
            #     cons += [z[2 * i_obs: 2 * (i_obs + 1), i_t] == sos1_code]

          for i_t in range(self.N-1):
            lambd = cp.Variable(4)
            # lambd = y[4*i_obs:4*(i_obs+1), i_t]
            cons += [sum(lambd) == 1]
            a = GrayCode()
            X = a.getGray(num_y)
            Xlist = []
            for i in range(2 ** num_y):
                Xlist += X[i]
            Xarr = list(map(int, Xlist))
            num_row = int(len(Xarr) / num_y)
            Mat = np.array(Xarr).reshape(num_row, num_y)
            binary_encoding = Mat[:self.n*2, :]

            for i in range(2*self.n):
                cons += [lambd[i] >= 0]
                cons += [lambd[i] <= 1]

            for j in range(0, num_y):
                lambda_sum1 = 0
                lambda_sum2 = 0
                for k in range(0, self.n*2):
                    if binary_encoding[k, j] == 1:
                        lambda_sum1 += lambd[k]
                    elif binary_encoding[k, j] == 0:
                        lambda_sum2 += lambd[k]
                    else:
                        print("Runtime_error: The binary_encoding entry can be only 0 or 1.")
                        break
                cons += [lambda_sum1 <= z[j+i_obs*num_y, i_t]]
                cons += [lambda_sum2 <= 1 - z[j+i_obs*num_y, i_t]]
            cons += [y[4 * i_obs:4 * (i_obs + 1), i_t] == lambd]

        # Region bounds
        for kk in range(self.N):
          for jj in range(self.n):
            cons += [self.posmin[jj] - x[jj,kk] <= 0]
            cons += [x[jj,kk] - self.posmax[jj] <= 0]

        # Velocity constraints
        for kk in range(self.N):
          for jj in range(self.n):
            cons += [self.velmin - x[self.n+jj,kk] <= 0]
            cons += [x[self.n+jj,kk] - self.velmax <= 0]

        # Control constraints
        for kk in range(self.N-1):
          cons += [cp.norm(u[:,kk]) <= self.umax]

        M = 1000. # big M value
        lqr_cost = 0.
        # l2-norm of lqr_cost
        for kk in range(self.N):
          lqr_cost += cp.quad_form(x[:,kk]-xg, self.Q)

        for kk in range(self.N-1):
          lqr_cost += cp.quad_form(u[:,kk], self.R)
        self.stl_prob = cp.Problem(cp.Minimize(lqr_cost), cons)


    def solve_stl(self, params, solver=cp.GUROBI, msk_param_dict=None,verbose = False):
        """High-level method to solve parameterized MICP.

        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy parameters to their values
        for p in self.sampled_params:  # 采样一个点的para, 每个para都传递给binprob来求解（bin中有x,u,y）#
            self.stl_prob_parameters[p].value = params[p]

        ## TODO(pculbertson): allow different sets of params to vary.

        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        if solver == cp.MOSEK:
            # See: https://docs.mosek.com/9.1/dotnetfusion/param-groups.html#doc-param-groups
            if not msk_param_dict:
                with open('../solver_config/mosek.yaml') as file:
                    msk_param_dict = yaml.load(file, Loader=yaml.FullLoader)

            self.stl_prob.solve(solver=solver, mosek_params=msk_param_dict, verbose=verbose)
        elif solver == cp.GUROBI:
            # rel_path = os.getcwd()
            # grb_param_dict = {}
            # with open(os.path.join(rel_path, 'solver_config/gurobi.yaml')) as file:
            with open('../solver_config/gurobi.yaml') as file:
                grb_param_dict = yaml.load(file, Loader=yaml.FullLoader)
                self.stl_prob.solve(solver=solver, **grb_param_dict, verbose=verbose)

        # solve_time = self.bin_prob.solver_stats.solve_time
        solve_time = self.stl_prob.solver_stats.solve_time
        x_star, u_star, y_star, z_star = None, None, None,None
        if self.stl_prob.status in ['optimal', 'optimal_inaccurate'] and self.stl_prob.status not in ['infeasible',
                                                                                                      'unbounded']:
            prob_success = True
            cost = self.stl_prob.value
            x_star = self.stl_prob_variables['x'].value
            u_star = self.stl_prob_variables['u'].value
            y_star = self.stl_prob_variables['y'].value.astype(int)
            z_star = self.stl_prob_variables['z'].value.astype(int)

        # Clear any saved params
        for p in self.sampled_params:
            self.stl_prob_parameters[p].value = None

        return prob_success, cost, solve_time, (x_star, u_star, y_star, z_star)
        # return prob_success, cost, solve_time, (x_star, u_star, y_star)

    def solve_micp(self, params, solver=cp.MOSEK, msk_param_dict=None, verbose = False):
        """High-level method to solve parameterized MICP.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy parameters to their values
        for p in self.sampled_params:#采样一个点的para, 每个para都传递给binprob来求解（bin中有x,u,y）#
            self.bin_prob_parameters[p].value = params[p]

        ## TODO(pculbertson): allow different sets of params to vary.
        
        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        if solver == cp.MOSEK:
            # See: https://docs.mosek.com/9.1/dotnetfusion/param-groups.html#doc-param-groups
            if not msk_param_dict:
              msk_param_dict = {}
              # with open(os.path.join(os.environ['CoCo'], 'solver_config/mosek.yaml')) as file:
              with open('../solver_config/mosek.yaml') as file:
                  msk_param_dict = yaml.load(file, Loader=yaml.FullLoader)

            self.bin_prob.solve(solver=solver, mosek_params=msk_param_dict)
        elif solver == cp.GUROBI:
            grb_param_dict = {}
            # with open(os.path.join(os.environ['CoCo'], 'solver_config/gurobi.yaml')) as file:
            with open('../solver_config/gurobi.yaml') as file:
                grb_param_dict = yaml.load(file, Loader=yaml.FullLoader)

            self.bin_prob.solve(solver=solver, **grb_param_dict,verbose=verbose)
        # solve_time = self.bin_prob.solver_stats.solve_time
        solve_time = self.bin_prob._solve_time

        x_star, u_star, y_star = None, None, None
        if self.bin_prob.status in ['optimal', 'optimal_inaccurate'] and self.bin_prob.status not in ['infeasible', 'unbounded']:
            prob_success = True
            cost = self.bin_prob.value
            x_star = self.bin_prob_variables['x'].value
            u_star = self.bin_prob_variables['u'].value
            y_star = self.bin_prob_variables['y'].value.astype(int)

        # Clear any saved params
        for p in self.sampled_params:
            self.bin_prob_parameters[p].value = None

        return prob_success, cost, solve_time, (x_star, u_star, y_star)

    def solve_pinned(self, params, strat, solver=cp.GUROBI, verbose = False):
        """High-level method to solve MICP with pinned params & integer values.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            strat: numpy integer array, corresponding to integer values for the
                desired strategy.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy params to their values
        for p in self.sampled_params:
            self.mlopt_prob_parameters[p].value = params[p]

        self.mlopt_prob_parameters['y'].value = strat

        ## TODO(pculbertson): allow different sets of params to vary.

        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        self.mlopt_prob.solve(solver=solver, verbose = verbose)

        solve_time = self.mlopt_prob.solver_stats.solve_time
        x_star, u_star, y_star = None, None, strat
        if self.mlopt_prob.status == 'optimal':
            prob_success = True
            cost = self.mlopt_prob.value
            x_star = self.mlopt_prob_variables['x'].value
            u_star = self.mlopt_prob_variables['u'].value

        # Clear any saved params
        for p in self.sampled_params:
            self.mlopt_prob_parameters[p].value = None
        self.mlopt_prob_parameters['y'].value = None

        return prob_success, cost, solve_time, (x_star, u_star, y_star)

    def tight_constraint(self, z):
        tight_index = []
        for i_obs in range(self.n_obs):
            for j in range(self.N - 1):
                x = z[2 * i_obs:2 * (i_obs + 1), j]  # 二进制列表
                x_int = int("".join(str(i) for i in x), 2)  # 转换为int
                z_int = graycode.gray_code_to_tc(x_int)
                z_int = z_int + 4*j
                tight_index.append(z_int)
        tight_matri = np.reshape(tight_index, (self.n_obs, self.N - 1))

        return tight_matri


    def stl_tight_constraint(self,z):
        tight_index = []
        for j in range(len(z)):
            x = z[j]  # 二进制列表
            x_int = int("".join(str(int(i)) for i in x), 2)  # 转换为int
            z_int = graycode.gray_code_to_tc(x_int)
            tight_index.append(z_int)
        return tight_index


    def which_M(self, x, obstacles, eq_tol=1e-5, ineq_tol=1e-5):
        """Method to check which big-M constraints are active.
        
        Args:
            x: numpy array of size [2*self.n, self.N], state trajectory.
            obstacles: numpy array of size [4, self.n_obs]
            eq_tol: tolerance for equality constraints, default of 1e-5.
            ineq_tol : tolerance for ineq. constraints, default of 1e-5.
            
        Returns:
            violations: list of which logical constraints are violated.
        """
        violations = [] # list of obstacle big-M violations

        for i_obs in range(self.n_obs):
            curr_violations = [] # violations for current obstacle
            for i_t in range(self.N-1):
                for i_dim in range(self.n):
                    o_min = obstacles[self.n*i_dim,i_obs]
                    if (x[i_dim,i_t+1] - o_min > ineq_tol):
                        curr_violations.append(self.n*i_dim + 2*self.n*i_t)

                    o_max = obstacles[self.n*i_dim+1,i_obs]
                    if (-x[i_dim,i_t+1]  + o_max > ineq_tol):
                        curr_violations.append(self.n*i_dim+1 + 2*self.n*i_t)
            curr_violations = list(set(curr_violations))
            curr_violations.sort()
            violations.append(curr_violations)
        return violations

    def construct_features(self, params, prob_features, ii_obs=None):
        """Helper function to construct feature vector from parameter vector.

        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            prob_features: list of strings, desired features for classifier.
            ii_obs: index of obstacle strategy being queried; appends one-hot
                encoding to end of feature vector
        """
        feature_vec = np.array([])

        ## TODO(pculbertson): make this not hardcoded
        x0, xg = params['x0'], params['xg']
        obstacles = params['obstacles']
        x0 = params['x0']


        for feature in prob_features:
            if feature == "x0":
                feature_vec = np.hstack((feature_vec, x0))
            elif feature == "xg":
                feature_vec = np.hstack((feature_vec, xg))
            elif feature == "obstacles":
                feature_vec = np.hstack((feature_vec, np.reshape(obstacles, (4*self.n_obs))))
            elif feature == "obstacles_map":
                continue
            else:
                print('Feature {} is unknown'.format(feature))

        # Append one-hot encoding to end
        if ii_obs is not None:
            one_hot = np.zeros(self.n_obs)
            one_hot[ii_obs] = 1.
            feature_vec = np.hstack((feature_vec, one_hot))

        return feature_vec
