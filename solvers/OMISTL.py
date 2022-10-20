import os

import cvxpy
import cvxpy as cp
import pickle
import numpy as np
import pdb
import time
import random
import sys
import itertools
import torch

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sigmoid
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# sys.path.insert(1, os.environ['CoCo'])
# sys.path.insert(1, os.path.join(os.environ['CoCo'], 'pytorch'))

from core import Problem, Solver
from pytorch.models import FFNet, CNNet

class OMISTL(Solver):
    def __init__(self, system, problem, prob_features, n_evals=2):
        """Constructor for CoCo FF class.

        Args:
            system: string for system name e.g. 'cartpole'
            problem: Problem object for system
            prob_features: list of features to be used
            n_evals: number of strategies attempted to be solved
        """
        super().__init__()
        self.system = system
        self.problem = problem
        self.prob_features = prob_features
        self.n_evals = n_evals

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None

        # training parameters
        self.training_params = {}
        self.training_params['TRAINING_ITERATIONS'] = int(1500)
        # self.training_params['BATCH_SIZE'] = 128
        self.training_params['BATCH_SIZE'] = 64
        self.training_params['CHECKPOINT_AFTER'] = int(1000)
        self.training_params['SAVEPOINT_AFTER'] = int(30000)
        # self.training_params['SAVEPOINT_AFTER'] = int(50)
        self.training_params['TEST_BATCH_SIZE'] = 32

    def construct_stl_strategies(self, train_data, n_features):
        """ Reads in training data and constructs strategy dictonary
            TODO(acauligi): be able to compute strategies for test-set too
        """
        self.strategy_dict = {}

        p_train = train_data[0]  #x0,obs,xg#
        # obs_train = p_train['obstacles']
        x_train = train_data[1]  #X#
        y_train = train_data[-2] #y的值，大M矩阵#
        z_train = train_data[-1] #SOS1 encoding#
        for k in p_train.keys():
            self.num_train = len(p_train[k])  #10000#

        Y = np.array(z_train[0])
        y_true = [int(x) for item in Y for x in item]
        self.n_z = len(y_true)
        self.n_features = n_features
        num_probs = self.num_train
        params = p_train       #x0,obs,xg
        self.Y = y_train       #vector
        self.Z = z_train       #SOS1
        self.depth = len(z_train[0])
        # self.n_y = int(self.Y[0].size / self.problem.n_obs)   #160/8=20，每个障碍在N内的整数var数量

        # self.z_shape = self.Z[0].shape
        self.features = np.zeros((num_probs, self.n_features))      #80000*4
        self.labels = np.zeros((num_probs, self.n_z+1), dtype=int) #10000*53,第一位储存strategy的index
        self.n_strategies = 0

        for ii in range(num_probs):
            # obs_strats = self.problem.tight_constraint(z_train[ii])
            try:
                obs_strats = self.problem.stl_tight_constraint(self.Z[ii], self.depth)
            except:
                continue
            # obs_strats = self.problem.which_M(x_train[ii], obs_train[ii]) #给定x移动轨迹和obs位置，return activte bigM的下标list，有n_obs行，每行为4*（T-1)中active index
            prob_params = {}
            for k in params:
                prob_params[k] = params[k][ii]

            Y = np.array(z_train[ii])
            y_true = [int(x) for item in Y for x in item]
            y_true = np.array(y_true)

            obs_strat = tuple(obs_strats)
            if obs_strat not in self.strategy_dict.keys():#strategy_dic的key是active下标，值是20+1，integer strategy 和 一位strategy index
                self.strategy_dict[obs_strat] = np.hstack((self.n_strategies, np.copy(y_true)))
                self.n_strategies += 1

            self.labels[ii] = self.strategy_dict[obs_strat]
            self.features[ii] = self.problem.construct_features(prob_params, self.prob_features)
        self.strategy_mat = np.unique(self.labels, axis=0)
        sos1_shape = []
        for i in range(self.depth):
            a = (len(self.Z[0][i]))
            sos1_shape.append(a)
        self.sos1_shape = sos1_shape

        vector_shape = []
        for i in range(self.depth):
            a = (len(self.Y[0][i]))
            vector_shape.append(a)
        self.vector_shape = vector_shape

    def construct_strategies(self, n_features, train_data):
        """ Reads in training data and constructs strategy dictonary
            TODO(acauligi): be able to compute strategies for test-set too
        """
        self.n_features = n_features
        self.strategy_dict = {}

        p_train = train_data[0]  #x0,obs,xg#
        obs_train = p_train['obstacles']
        x_train = train_data[1]  #X#
        y_train = train_data[3] #y的值，大M矩阵#
        z_train = train_data[4] #SOS1 encoding#
        for k in p_train.keys():
            self.num_train = len(p_train[k])  #10000#

        ## TODO(acauligi): add support for combining p_train & p_test correctly
        ## to be able to generate strategies for train and test params here
        # p_test = None
        # x_test = None
        # y_test = np.empty((*y_train.shape[:-1], 0))   # Assumes y_train is 3D tensor
        # if test_data:
        #   p_test, y_test = test_data[:2]
        #   for k in p_test.keys():
        #     self.num_test = len(p_test[k])
        # num_probs = self.num_train + self.num_test
        # self.Y = np.dstack((Y_train, Y_test))         # Stack Y_train and Y_test along dim=2
        num_probs = self.num_train
        params = p_train       #x0,obs,xg
        self.Y = y_train       #bigM
        self.Z = z_train

        self.n_y = int(self.Y[0].size / self.problem.n_obs)   #160/8=20，每个障碍在N内的整数var数量
        self.y_shape = self.Y[0].shape  #32*5
        self.features = np.zeros((self.problem.n_obs*num_probs, self.n_features))      #80000*4
        self.cnn_features = None
        self.cnn_features_idx = None
        self.labels = np.zeros((self.problem.n_obs*num_probs, 1+self.n_y), dtype=int) #80000*21,第一位储存strategy的index
        self.n_strategies = 0


        for ii in range(num_probs):
            obs_strats = self.problem.tight_constraint(z_train[ii])
            # obs_strats = self.problem.which_M(x_train[ii], obs_train[ii]) #给定x移动轨迹和obs位置，return activte bigM的下标list，有n_obs行，每行为4*（T-1)中active index

            prob_params = {}
            for k in params:
                prob_params[k] = params[k][ii]

            for ii_obs in range(self.problem.n_obs):
                # TODO(acauligi): check if transpose necessary with new pickle save format for Y
                y_true = np.reshape(self.Y[ii, 4*ii_obs:4*(ii_obs+1),:], (self.n_y))#把单个obs的4x5的bigM binary矩阵转化成20x1，作为lable
                obs_strat = tuple(obs_strats[ii_obs])   #储存单个obs中active bigM的下标

                if obs_strat not in self.strategy_dict.keys():#strategy_dic的key是active下标，值是20+1，integer strategy 和 一位strategy index
                    self.strategy_dict[obs_strat] = np.hstack((self.n_strategies, np.copy(y_true))) #第0位是strategy的index，跟20位的integer合并起来#
                    self.n_strategies += 1

                self.labels[ii*self.problem.n_obs+ii_obs] = self.strategy_dict[obs_strat]#lables:80000*21, 1 onehot + 4*(N-1)bigM#
                #   feature: if "obstacle" in prob_feat,创建第ii个obs的feature vector#
                #   params: x0:, obs:, xg:#
                #   prob_features: list of strings, desired features for classifier#
                # ii_obs: index of obstacle strategy being queried; appends one-hot encoding to end of feature vector#
                self.features[ii*self.problem.n_obs+ii_obs] = self.problem.construct_features(prob_params, self.prob_features, \
                        ii_obs=ii_obs if 'obstacles' in self.prob_features else None)#创建第ii个obstacle的特征向量，对应的onehot为1#

        self.strategy_mat = np.unique(self.labels, axis=0)


    def setup_network(self, depth=3, neurons=128, device_id=0):
        if device_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(device_id))

        ff_shape = []
        for ii in range(depth):
            ff_shape.append(neurons)  #每一层添加神经元#

        ff_shape.append(self.n_strategies) #最后一层对应n个strategies
        ff_shape.insert(0, self.n_features)  #[44, 128, 128, 128, 434]
        self.model = FFNet(ff_shape, activation=torch.nn.ReLU()).to(device=self.device)#把模型分配到device上#

        # file names for PyTorch models
        model_fn = 'models/{}_horizon_{}_obs_{}.pt'
        model_fn = os.path.join('../', model_fn)
        self.model_fn = model_fn.format(self.system, self.problem.N,self.problem.n_obs)

    def load_network(self, fn_classifier_model):
        if os.path.exists(fn_classifier_model):
            print('Loading presaved classifier model from {}'.format(fn_classifier_model))
            self.model.load_state_dict(torch.load(fn_classifier_model))
            self.model_fn = fn_classifier_model
            return True
        else:
            print('No existing model! Starting training model.')
            return False

    def train_stl(self, train_data=None, verbose=True):
        # grab training params
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

        model = self.model
        model.to(device=self.device)

        X = self.features[:self.num_train] #X是44列特征向量,[x0(4),obs(4x8),onehot(8)]#

        Y = self.labels[:self.num_train,0] #Y是21列特征,4x5的interger和1位onehot#

        training_loss = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.00001)

        itr = 1
        for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            rand_idx = list(np.arange(0,X.shape[0]-1))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii,idx in enumerate(indices):
                # zero the parameter gradients
                opt.zero_grad()

                ff_inputs = Variable(torch.from_numpy(X[idx,:])).float().to(device=self.device)
                labels = Variable(torch.from_numpy(Y[idx])).long().to(device=self.device)

                # forward + backward + optimize
                outputs = None
                outputs = model(ff_inputs) #input是tensor[xg,obs,onehot],output是长度为n_strategy的预测张量#

                loss = training_loss(outputs, labels).float().to(device=self.device)
                class_guesses = torch.argmax(outputs,1)  #tensor([3], device='cuda:0', grad_fn=<NotImplemented>)
                accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
                opt.step()

                # print statistics\n",
                running_loss += loss.item()
                if itr % CHECKPOINT_AFTER == 0:
                    rand_idx = list(np.arange(0,X.shape[0]-1))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    ff_inputs = Variable(torch.from_numpy(X[test_inds,:])).float().to(device=self.device)
                    labels = Variable(torch.from_numpy(Y[test_inds])).long().to(device=self.device)

                    # forward + backward + optimize

                    outputs = model(ff_inputs)

                    loss = training_loss(outputs, labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,1)
                    accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                    verbose and print("loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.state_dict(), self.model_fn)
                    verbose and print('Saved model at {}'.format(self.model_fn))
                    # writer.add_scalar('Loss/train', running_loss, epoch)

                itr += 1
            verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        torch.save(model.state_dict(), self.model_fn)
        print('Saved model at {}'.format(self.model_fn))
        print('Done training')

    def train(self, train_data=None, verbose=True):
        # grab training params
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

        model = self.model
        model.to(device=self.device)

        X = self.features[:self.problem.n_obs*self.num_train] #X是44列特征向量,[x0(4),obs(4x8),onehot(8)]#
        X_cnn = None
        if 'obstacles_map' in self.prob_features:
            # X_cnn = self.cnn_features[:self.problem.n_obs*self.num_train]
            # TODO(acauligi)
            params = train_data[0]
            X_cnn = np.zeros((BATCH_SIZE, 3,self.problem.H,self.problem.W))
        Y = self.labels[:self.problem.n_obs*self.num_train,0] #Y是21列特征,4x5的interger和1位onehot#

        training_loss = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.00001)

        itr = 1
        for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            rand_idx = list(np.arange(0,X.shape[0]-1))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii,idx in enumerate(indices):
                # zero the parameter gradients
                opt.zero_grad()

                ff_inputs = Variable(torch.from_numpy(X[idx,:])).float().to(device=self.device)
                labels = Variable(torch.from_numpy(Y[idx])).long().to(device=self.device)

                # forward + backward + optimize
                outputs = None
                if 'obstacles_map' in self.prob_features:
                    X_cnn = np.zeros((len(idx), 3,self.problem.H,self.problem.W))
                    for idx_ii, idx_val in enumerate(idx):
                        prob_params = {}
                        for k in params:
                            prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]  #从样本下标中取出#
                        X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, \
                        ii_obs=self.cnn_features_idx[idx_val][1])#cnn_features_idx[idx_val][1]#是n_obs下标

                    cnn_inputs = Variable(torch.from_numpy(X_cnn)).float().to(device=self.device)
                    outputs = model(cnn_inputs, ff_inputs)
                else:
                    outputs = model(ff_inputs) #input是tensor[xg,obs,onehot],output是长度为n_strategy的预测张量#

                loss = training_loss(outputs, labels).float().to(device=self.device)
                class_guesses = torch.argmax(outputs,1)  #tensor([3], device='cuda:0', grad_fn=<NotImplemented>)
                accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
                opt.step()

                # print statistics\n",
                running_loss += loss.item()
                if itr % CHECKPOINT_AFTER == 0:
                    rand_idx = list(np.arange(0,X.shape[0]-1))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    ff_inputs = Variable(torch.from_numpy(X[test_inds,:])).float().to(device=self.device)
                    labels = Variable(torch.from_numpy(Y[test_inds])).long().to(device=self.device)

                    # forward + backward + optimize
                    if type(model) is CNNet:
                        X_cnn = np.zeros((len(test_inds), 3,self.problem.H,self.problem.W))
                        for idx_ii, idx_val in enumerate(test_inds):
                            prob_params = {}
                            for k in params:
                                prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                            X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=self.cnn_features_idx[idx_val][1])
                        cnn_inputs = Variable(torch.from_numpy(X_cnn)).float().to(device=self.device)
                        # cnn_inputs = Variable(torch.from_numpy(X_cnn[test_inds,:])).float().to(device=self.device)
                        outputs = model(cnn_inputs, ff_inputs)
                    else:
                        outputs = model(ff_inputs)

                    loss = training_loss(outputs, labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,1)
                    accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                    verbose and print("loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.state_dict(), self.model_fn)
                    verbose and print('Saved model at {}'.format(self.model_fn))
                    # writer.add_scalar('Loss/train', running_loss, epoch)
                itr += 1
            verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        torch.save(model.state_dict(), self.model_fn)
        print('Saved model at {}'.format(self.model_fn))
        print('Done training')

    def forward(self, prob_params, solver=cp.MOSEK, max_evals=16,verbose = False):
        self.model.to(device=torch.device('cpu'))
        y_guesses = np.zeros((self.n_evals ** self.problem.n_obs, self.n_y), dtype=int)

        # Compute forward pass for each obstacle and save the top
        # n_eval's scoring strategies in ind_max
        total_time = 0.  # start timing forward passes of network
        features = np.zeros((self.problem.n_obs, self.n_features))
        for ii_obs in range(self.problem.n_obs):
            features[ii_obs] = self.problem.construct_features(prob_params, self.prob_features, ii_obs=ii_obs)
        inpt = Variable(torch.from_numpy(features)).unsqueeze(0).float()
        t0 = time.time()
        with torch.no_grad():
            scores = self.model(inpt).cpu().detach().numpy()[:].squeeze(0)  # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        torch.cuda.synchronize()
        # total_time += (time.time()-t0)
        ind_max = np.argsort(scores, axis=1)[:, -self.n_evals:][:, ::-1]  # the index of top n_eval best strategies

        # Loop through strategy dictionary once
        # Save ii'th stratey in obs_strats dictionary
        obs_strats = {}
        uniq_idxs = np.unique(ind_max)  # 在(n_obs*n_eval)中返回唯一的strategy index list#
        # for ii,idx in enumerate(uniq_idxs):
        #     for jj in range(self.labels.shape[0]):
        #         # first index of training label is that strategy's idx
        #         label = self.labels[jj]
        #         if label[0] == idx: # go through the idx of n_strategy for 80000 samples，如果idx是lable的第一位的话，把这个lable后面的pin值给obs_strats
        #             # remainder of training label is that strategy's binary pin
        #             obs_strats[idx] = label[1:]
        for ii, idx in enumerate(uniq_idxs):
            obs_strats[idx] = self.strategy_mat[idx, 1:]
        total_time += (time.time() - t0)

        # Generate Cartesian product of strategy combinations
        vv = [np.arange(0, self.n_evals) for _ in range(self.problem.n_obs)]
        strategy_tuples = list(itertools.product(*vv))  # 创建combaination list, 大小为n_eval**n_obs#

        # Sample from candidate strategy tuples based on "better" combinations
        probs_str = [1. / (np.sum(st) + 1.) for st in strategy_tuples]  # lower sum(st) values --> better
        probs_str = probs_str / np.sum(probs_str)
        str_idxs = np.random.choice(np.arange(0, len(strategy_tuples)), max_evals,
                                    p=probs_str)  # pick number of max_evals strategies combanation index

        # Manually add top-scoring strategy tuples
        if 0 in str_idxs:
            str_idxs = np.unique(np.insert(str_idxs, 0, 0))  # 随到0全置0#
        else:
            str_idxs = np.insert(str_idxs, 0, 0)[:-1]
        strategy_tuples = [strategy_tuples[ii] for ii in str_idxs]

        prob_success, cost, n_evals, optvals = False, np.Inf, max_evals, None
        for ii, str_tuple in enumerate(strategy_tuples):
            y_guess = -np.ones((4 * self.problem.n_obs, self.problem.N - 1))
            for ii_obs in range(self.problem.n_obs):
                # rows of ind_max correspond to ii_obs, column to desired strategy
                y_obs = obs_strats[ind_max[ii_obs, str_tuple[ii_obs]]]  # 预测了最优策略的下标ind_max，每个障碍的integer用该策略赋值#
                y_guess[4 * ii_obs:4 * (ii_obs + 1)] = np.reshape(y_obs, (4, self.problem.N - 1))  # 输出所需的shape(32*5)对应的integer策略#
            if (y_guess < 0).any():
                print("Strategy was not correctly found!")
                return False, np.Inf, total_time, n_evals, optvals
            self.y_guess = y_guess
            prob_success, cost, solve_time, optvals = self.problem.solve_pinned(prob_params, y_guess, solver=solver, verbose=verbose)
            total_time += solve_time
            n_evals = ii + 1
            if prob_success:
                break

        return prob_success, cost, total_time, n_evals, optvals, y_guess

    def Predict(self, prob_params, solver=cp.MOSEK, max_evals=16, verbose=True):
        self.model.to(device=torch.device('cpu'))

        # Compute forward pass for each obstacle and save the top
        # n_eval's scoring strategies in ind_max
        total_time = 0.   # start timing forward passes of network
        features = self.problem.construct_features(prob_params, self.prob_features)
        inpt = Variable(torch.from_numpy(features)).unsqueeze(0).float()
        t0 = time.time()
        with torch.no_grad():
            scores = self.model(inpt).cpu().detach().numpy()[:].squeeze(0)#从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        torch.cuda.synchronize()
        # total_time += (time.time()-t0)
        ind_max = np.argsort(scores)[-max_evals:][::-1] # the index of top n_eval best strategies

        # Loop through strategy dictionary once
        # Save ii'th stratey in obs_strats dictionary
        obs_strats = {}
        uniq_idxs = np.unique(ind_max) #在(n_obs*n_eval)中返回唯一的strategy index list#
        for ii, idx in enumerate(uniq_idxs):
            obs_strats[idx] = self.strategy_mat[idx, 1:]
        total_time += (time.time() - t0)
        y_guess_mat = []
        for i in range(max_evals):
            y_guess = obs_strats[ind_max[i]]
            y_guess_mat.append(y_guess)

        for n_eval in range(max_evals):
            sos1 = self.recover_sos1(y_guess_mat[n_eval])
            try:
                recover_solution = self.recover_solution(sos1)
            except:
                continue
            self.problem.init_pred_problem(recover_solution)
            prob_success, cost, solve_time, optvals = self.problem.solve_pred(prob_params, solver=cp.GUROBI, verbose=verbose)
            if prob_success:
                print('prediction is success:' + str(prob_success))
                break
            else:
                if n_eval == max_evals:
                    prob_success = 'False'
                    print('prediction is infeasible')
                    break
        total_time += solve_time

        return prob_success, cost, total_time, recover_solution, optvals

    def recover_sos1(self,y_guess):
        integer_mat = []
        start = 0
        for i in range(len(self.sos1_shape)):
            end = start + self.sos1_shape[i]
            integer_mat.append(y_guess[start:end])
            start = end
        return integer_mat

    def recover_solution(self, integer):
        solution = []
        for i in range(self.depth):
            sos1 = integer[i]
            zero_pos = ''.join([str(int(x)) for x in sos1])
            zero_pos = int(self.gray2binary(zero_pos),2)
            vector = np.zeros(self.vector_shape[i])
            vector[zero_pos] =1
            solution.append(vector)
        return solution

    def gray2binary(self,n):
        n = int(n, 2)
        mask = n
        while mask:
            mask >>= 1
            n ^= mask
        return bin(n)[2:].zfill(4)
