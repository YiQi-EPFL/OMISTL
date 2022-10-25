import os
import cvxpy as cp
import yaml
import pickle
import numpy as np
import sys
import pdb
from sos1 import addsos1constraint
from stlpy.STL.predicate import LinearPredicate,NonlinearPredicate

import graycode
sys.path.insert(1, os.environ['CoCo'])

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from core import Problem


class STL_planner(Problem):
    """Class to setup + solve free-flyer problems."""

    def     __init__(self, config=None, spec=None, solver=cp.MOSEK):
        """Constructor for FreeFlyer class.

        Args:
            config: full path to solver_config file. if None, load default solver_config.
            solver: solver object to be used by cvxpy
        """
        super().__init__()

        ## TODO(pculbertson): allow different sets of params to vary.
        if config is None:  # use default solver_config
            relative_path = os.path.dirname(os.path.abspath(__file__))
            config = relative_path + '/solver_config/default.p'
        self.spec = spec
        config_file = open(config, "rb")
        configs = pickle.load(config_file)
        prob_params = configs[1]
        self.sampled_params = configs[2]
        config_file.close()
        self.init_problem(prob_params)

    def init_problem(self, prob_params):
        # setup problem params
        self.n = 2
        self.m = 2

        self.N, self.Ak, self.Bk,self.Ck, self.Dk, self.Q, self.R, self.n_obs, \
          self.posmin, self.posmax, self.velmin, self.velmax, \
          self.umin, self.umax = prob_params

        self.H = 64
        self.W = int(self.posmax[0] / self.posmax[1] * self.H)
        self.H, self.W = 32, 32
        self.init_stl_problem()


    def init_stl_problem(self, typical_encoding = False):
        self.cons = []
        p = self.Ck.shape[0]
        # Variables
        # x = cp.Variable((2 * self.n, self.N))  # state
        self.varlist = {}
        x = cp.Variable((2*self.n, self.N))
        u = cp.Variable((self.m, self.N))  # control
        y = cp.Variable((p, self.N))
        self.y = y
        rho = cp.Variable(1)
        self.rho = rho
        self.cons += [rho >= 0.0]

        self.varlist['x'] = x
        self.varlist['u'] = u
        self.varlist['y'] = y
        self.varlist['rho'] = rho

        self.stl_prob_variables = {'x': x, 'u': u, 'y':y, 'rho':rho}

        # Parameters
        x0 = cp.Parameter(2 * self.n)
        # xg = cp.Parameter(2 * self.n)

        self.stl_prob_parameters = {'x0': x0}
                                        #, 'xg': xg
                                      # 'obstacles': obstacles}

        # Dynamics constraints
        self.cons += [x[:, 0] == x0]

        # for ii in range(self.N - 1):
        #     self.cons += [x[:, ii + 1] - (self.Ak @ x[:, ii] + self.Bk @ u[:, ii]) == np.zeros(4)]

        for ii in range(self.N - 1):
            self.cons += [x[:, ii + 1] == (self.Ak @ x[:, ii] + self.Bk @ u[:, ii])]
            self.cons += [y[:, ii] == self.Ck@x[:, ii] + self.Dk@u[:, ii]]
        self.cons += [y[:, self.N - 1] == self.Ck @ x[:, self.N - 1] + self.Dk @ u[:, self.N - 1]]


        # Region bounds
        for kk in range(self.N):
            self.cons += [self.posmin - x[:, kk] <= 0]
            self.cons += [x[:, kk] <= self.posmax]

        # Control constraints
        for kk in range(self.N - 1):
            self.cons += [self.umin <= u[:, kk]]
            self.cons += [u[:, kk] <= self.umax]

        #set cost function
        stl_cost = 0.0
        # l2-norm of lqr_cost
        stl_cost += cp.quad_form(x[:, 0], self.Q) + cp.quad_form(u[:, 0], self.R)
        for t in range(1, self.N):
            stl_cost += cp.quad_form(x[:, t], self.Q) + cp.quad_form(u[:, t], self.R)
        stl_cost -= -1 * self.rho

        ##Add STL constraints
        # z_spec = cp.Variable(1, boolean=True)
        z_spec = cp.Variable(1)
        # z_spec = cp.Variable(1)
        self.cons += [z_spec == 1]
        self.varlist['z_spec'] = z_spec
        self.varstl_list = []
        self.integer = []
        self.vector = []
        if not typical_encoding:
            self.AddSOS1SubformulaConstraints(self.spec, z_spec, 0)
        else:
            self.AddSubformulaConstraints(self.spec, z_spec, 0)
        self.stl_prob = cp.Problem(cp.Minimize(stl_cost), self.cons)

    def solve_stl(self, params, solver=cp.GUROBI, msk_param_dict=None, verbose = True):
        """High-level method to solve parameterized MICP.

        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy parameters to their values
        for p in self.sampled_params:
            self.stl_prob_parameters[p].value = params[p]

        ## TODO(pculbertson): allow different sets of params to vary.

        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        if solver == cp.MOSEK:
            # See: https://docs.mosek.com/9.1/dotnetfusion/param-groups.html#doc-param-groups
            if not msk_param_dict:
                msk_param_dict = {}
                path = '../../solver_config/mosek.yaml'
                with open(path) as file:
                    msk_param_dict = yaml.load(file, Loader=yaml.FullLoader)

            self.stl_prob.solve(solver=solver, mosek_params=msk_param_dict, verbose=verbose)
        elif solver == cp.GUROBI:
            rel_path = os.getcwd()
            grb_param_dict = {}
            path = '../../solver_config/gurobi.yaml'
            with open(path) as file:
                grb_param_dict = yaml.load(file, Loader=yaml.FullLoader)
                self.stl_prob.solve(solver=solver, **grb_param_dict, verbose=verbose)

        # solve_time = self.bin_prob.solver_stats.solve_time
        solve_time = self.stl_prob.solver_stats.solve_time
        x_star, u_star, y_star = None, None, None
        vector_mat = []
        integer_mat = []
        self.n_features = 0
        if self.stl_prob.status in ['optimal', 'optimal_inaccurate'] and self.stl_prob.status not in ['infeasible',
                                                                                                      'unbounded']:
            prob_success = True
            cost = self.stl_prob.value
            x_star = self.stl_prob_variables['x'].value
            u_star = self.stl_prob_variables['u'].value
            y_star = self.stl_prob_variables['y'].value.astype(int)
            vector = self.vector
            integer = self.integer
            self.cons = None

            for i in range(len(vector)):
                vector_mat.append(vector[i].value)
            self.vector_mat = vector_mat

            for i in range(len(integer)):
                integer_mat.append(integer[i].value)
            self.integer_mat = integer_mat

            Y = np.array(self.integer_mat)
            y_true = [int(x) for item in Y for x in item]
            self.n_features = len(y_true)

        # Clear any saved params
        for p in self.sampled_params:
            self.stl_prob_parameters[p].value = None

        return prob_success, cost, solve_time, integer_mat, vector_mat, (x_star, u_star, y_star)

    def init_pred_problem(self, vector_mat):
        self.cons = []
        p = self.Ck.shape[0]

        # Variables
        # x = cp.Variable((2 * self.n, self.N))  # state
        self.varlist = {}
        x = cp.Variable((2 * self.n, self.N))
        u = cp.Variable((self.m, self.N))  # control
        y = cp.Variable((p, self.N))
        self.y = y
        rho = cp.Variable(1)
        self.rho = rho
        self.cons += [rho >= 0.0]

        self.varlist['x'] = x
        self.varlist['u'] = u
        self.varlist['y'] = y
        self.varlist['rho'] = rho

        self.pred_prob_variables = {'x': x, 'u': u, 'y': y, 'rho': rho}

        # Parameters
        x0 = cp.Parameter(2 * self.n)

        self.pred_prob_parameters = {'x0': x0}

        # Dynamics constraints
        self.cons += [x[:, 0] == x0]

        # for ii in range(self.N - 1):
        #     self.cons += [x[:, ii + 1] - (self.Ak @ x[:, ii] + self.Bk @ u[:, ii]) == np.zeros(4)]

        for ii in range(self.N - 1):
            self.cons += [x[:, ii + 1] == (self.Ak @ x[:, ii] + self.Bk @ u[:, ii])]
            self.cons += [y[:, ii] == self.Ck @ x[:, ii] + self.Dk @ u[:, ii]]
        self.cons += [y[:, self.N - 1] == self.Ck @ x[:, self.N - 1] + self.Dk @ u[:, self.N - 1]]

        # Region bounds
        for kk in range(self.N):
            self.cons += [self.posmin - x[:, kk] <= 0]
            self.cons += [x[:, kk] <= self.posmax]

        # Control constraints
        for kk in range(self.N - 1):
            self.cons += [self.umin <= u[:, kk]]
            self.cons += [u[:, kk] <= self.umax]

        # set cost function
        stl_cost = 0.0
        # l2-norm of lqr_cost
        stl_cost += cp.quad_form(x[:, 0], self.Q) + cp.quad_form(u[:, 0], self.R)
        for t in range(1, self.N):
            stl_cost += cp.quad_form(x[:, t], self.Q) + cp.quad_form(u[:, t], self.R)
        stl_cost -= -1 * self.rho

        z_spec = cp.Variable(1)
        self.cons += [z_spec == 1]
        self.varlist['z_spec'] = z_spec
        self.varstl_list = []
        self.number_or = 0
        self.AddPredictedConstraints(self.spec, z_spec, 0, vector_mat)
        self.pred_prob = cp.Problem(cp.Minimize(stl_cost), self.cons)

    def AddSubformulaConstraints(self, formula, z, t):
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, LinearPredicate):
            # a.T*y - b + (1-z)*M >= rho
            M = 200
            # z = cp.Variable(1, boolean=True, name='leaf')
            self.cons += [formula.a.T @ self.y[:, t] - formula.b + (1-z)*M >= self.rho]
            self.varstl_list.append(z)

        elif isinstance(formula, NonlinearPredicate):
            raise TypeError("Mixed integer programming does not support nonlinear predicates")

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = cp.Variable(1)
                    self.varstl_list.append(z_sub)
                    t_sub = formula.timesteps[i]   # the timestep at which this formula
                                                   # should hold

                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)
                    self.cons += [z <= z_sub ]

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = cp.Variable(1, boolean=True)
                    # z_sub = cp.Variable(1)
                    self.varstl_list.append(z_sub)
                    # z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                    # z_sub = self.model.addMVar(1, vtype=GRB.BINARY)
                    z_subs.append(z_sub)
                    t_sub = formula.timesteps[i]
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub)
                self.cons += [z <= sum(z_subs)]

    def AddSOS1SubformulaConstraints(self, formula, z, t):
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, LinearPredicate):
            # a.T*y - b + (1-z)*M >= rho
            M = 200
            # z = cp.Variable(1, boolean=True, name='leaf')
            self.cons += [formula.a.T @ self.y[:, t] - formula.b + (1 - z) * M >= self.rho]
            self.varstl_list.append(z)

        else:
            nz = len(formula.subformula_list)
            if formula.combination_type == "and":
                z_subs = cp.Variable(nz)
                for i in range(nz):
                    self.cons += [z <= z_subs[i]]#and加一个新的z_zub,所有
            else:  # combination_type == "or":
                lambda_, y, consos1 = addsos1constraint(nz + 1)
                z_subs = lambda_[1:][np.newaxis].T
                self.cons += consos1
                self.cons += [1-z == lambda_[0]]
                self.integer.append(y)
                self.vector.append(lambda_)

            for i, subformula in enumerate(formula.subformula_list):
                t_sub = formula.timesteps[i]
                self.AddSOS1SubformulaConstraints(subformula, z_subs[i], t+t_sub)

    def AddPredictedConstraints(self, formula, z, t, vector_mat, j=0):
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, LinearPredicate):
            # a.T*y - b + (1-z)*M >= rho
            M = 200
            self.cons += [formula.a.T @ self.y[:, t] - formula.b + (1 - z) * M >= self.rho]
        else:
            nz = len(formula.subformula_list)
            if formula.combination_type == "and":
                # z_subs = np.ones(nz)
                z_subs = cp.Variable(nz)
                for i in range(nz):
                    self.cons += [z <= z_subs[i]]  # and加一个新的z_zub,所有
                for i, subformula in enumerate(formula.subformula_list):
                    t_sub = formula.timesteps[i]
                    self.AddPredictedConstraints(subformula, z_subs[i], t + t_sub, vector_mat, self.number_or)
            else:  # combination_type == "or":
                z_subs = vector_mat[j][1:]
                self.number_or += 1
                for i, subformula in enumerate(formula.subformula_list):
                    t_sub = formula.timesteps[i]
                    if z_subs[i] != 0:
                        self.AddPredictedConstraints(subformula, z_subs[i], t + t_sub, vector_mat, self.number_or)

    def solve_pred(self, params, solver=cp.GUROBI, msk_param_dict=None, verbose = True):
        """High-level method to solve parameterized MICP.

        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy parameters to their values
        for p in self.sampled_params:  # 采样一个点的para, 每个para都传递给binprob来求解（bin中有x,u,y）#
            self.pred_prob_parameters[p].value = params[p]

        ## TODO(pculbertson): allow different sets of params to vary.

        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        if solver == cp.MOSEK:
            # See: https://docs.mosek.com/9.1/dotnetfusion/param-groups.html#doc-param-groups
            if not msk_param_dict:
                msk_param_dict = {}
                path = '../../solver_config/mosek.yaml'
                with open(path) as file:
                    msk_param_dict = yaml.load(file, Loader=yaml.FullLoader)

            self.pred_prob.solve(solver=solver, mosek_params=msk_param_dict, verbose=verbose)
        elif solver == cp.GUROBI:
            rel_path = os.getcwd()
            grb_param_dict = {}
            path = '../../solver_config/gurobi.yaml'
            with open(path) as file:
                grb_param_dict = yaml.load(file, Loader=yaml.FullLoader)
                self.pred_prob.solve(solver=solver, **grb_param_dict, verbose=verbose)

        # solve_time = self.bin_prob.solver_stats.solve_time
        solve_time = self.pred_prob.solver_stats.solve_time
        x_star, u_star, y_star = None, None, None
        if self.pred_prob.status in ['optimal', 'optimal_inaccurate'] and self.stl_prob.status not in ['infeasible',
                                                                                                      'unbounded']:
            prob_success = True
            cost = self.pred_prob.value
            x_star = self.pred_prob_variables['x'].value
            u_star = self.pred_prob_variables['u'].value
            y_star = self.pred_prob_variables['y'].value.astype(int)

        # Clear any saved params
        for p in self.sampled_params:
            self.stl_prob_parameters[p].value = None
        self.cons = None
        self.integer = None
        self.vector = None

        return prob_success, cost, solve_time, (x_star, u_star, y_star)

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
        violations = []  # list of obstacle big-M violations

        for i_obs in range(self.n_obs):
            curr_violations = []  # violations for current obstacle
            for i_t in range(self.N - 1):
                for i_dim in range(self.n):
                    o_min = obstacles[self.n * i_dim, i_obs]
                    if (x[i_dim, i_t + 1] - o_min > ineq_tol):
                        curr_violations.append(self.n * i_dim + 2 * self.n * i_t)

                    o_max = obstacles[self.n * i_dim + 1, i_obs]
                    if (-x[i_dim, i_t + 1] + o_max > ineq_tol):
                        curr_violations.append(self.n * i_dim + 1 + 2 * self.n * i_t)
            curr_violations = list(set(curr_violations))
            curr_violations.sort()
            violations.append(curr_violations)
        return violations

    def stl_tight_constraint(self, z, depth):
        tight_index = []
        for indx in range(depth):
            x = z[indx]  # 二进制列表
            x_int = int("".join(str(int(i)) for i in x), 2)  # 转换为int
            z_int = graycode.gray_code_to_tc(x_int)
            tight_index.append(z_int)
        return tight_index

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

        # x0, xg = params['x0'], params['xg']
        # obstacles = params['obstacles']
        x0 = params['x0']


        for feature in prob_features:
            if feature == "x0":
                feature_vec = np.hstack((feature_vec, x0))
            # elif feature == "xg":
            #     feature_vec = np.hstack((feature_vec, xg))
            # elif feature == "obstacles":
            #     feature_vec = np.hstack((feature_vec, np.reshape(obstacles, (4 * self.n_obs))))
            # elif feature == "obstacles_map":
            #     continue
            else:
                print('Feature {} is unknown'.format(feature))

        return feature_vec
