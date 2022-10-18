# from para_stl import paraset
import os
import pickle
from stlpy.benchmark.reach_avoid import ReachAvoid
from stlpy.benchmark.random_multitarget import RandomMultitarget
import numpy as np
from stl_planner import FreeFlyer
import cvxpy as cp
from stlpy.benchmark.avoid_obstacles import Obstacles_Avoid
from stlpy.STL.predicate import LinearPredicate, NonlinearPredicate


#
N=10
x0 = [0.1,0.1,0,0]
# vmax = 0.5
prob_params= {}
prob_params['x0'] = x0
# Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
# R = 1e-1*np.eye(2)
#
posmin = np.array([0.0, 0.0, -1.0, -1.0])
posmax =np.array([3.5,2.5, 1.0, 1.0])
# umax = np.array([0.2, 0.2])

goal_bounds = (2.90, 3.25, 2.00, 2.25)     # (xmin, xmax, ymin, ymax)
obstacle_bounds = (1.25, 1.75, 0.20, 1.00)


scenario = ReachAvoid(goal_bounds, obstacle_bounds, N-1)
spec = scenario.GetSpecification()
spec.simplify()
spec_name= 'ReachAvoid'

dataset_name = '{}_horizon_{}'.format(spec_name, N)

#加载设置文件地址，solver_config = [dataset_name, prob_params, sampled_params]#
relative_path = os.getcwd()
config_fn = os.path.join(relative_path, 'config', dataset_name+'.p')

config_file = open(config_fn, 'rb')
config = pickle.load(config_file)

dataset_name = config[0]
config_file.close()#读取设置#

prob = FreeFlyer(config=config_fn, spec=spec)

prob_success, cost, solve_time,integer, vector, optvals =prob.solve_stl(prob_params,solver=cp.MOSEK)
# prob_success, cost, solve_time, optvals =prob.solve_pred(prob_params,solver=cp.MOSEK)

print(prob_success)
print(solve_time)



import matplotlib.pyplot as plt
Xopt = optvals[0]

goal_bounds = list(goal_bounds)    # (xmin, xmax, ymin, ymax)
obstacle_bounds = list(scenario.obstacle_bounds)


goal  = plt.Rectangle((goal_bounds[0], goal_bounds[2]), \
                                  goal_bounds[1]-goal_bounds[0], goal_bounds[3]-goal_bounds[2], \
                                 fc='white', ec='black')
obstacle = plt.Rectangle((obstacle_bounds[0], obstacle_bounds[2]), \
                                  obstacle_bounds[1]-obstacle_bounds[0], obstacle_bounds[3]-obstacle_bounds[2], \
                                 fc='white', ec='black')

plt.gca().add_patch(obstacle)
plt.gca().add_patch(goal)

plt.axis('scaled')


x0 = prob_params['x0']
circle = plt.Circle((x0[0],x0[1]), 0.04, fc='red',ec="red")
plt.gca().add_patch(circle)

#blue line is network prediction
# plt.plot(xg[0],xg[1],'sr')
# plt.quiver(Xopt[0,:], Xopt[1,:], Xopt[2,:], Xopt[3,:])#用箭头表示#
for jj in range(N):
    circle = plt.Circle((Xopt[0,jj],Xopt[1,jj]), 0.04, fc='black',ec="black")
    plt.gca().add_patch(circle)


posmin = np.zeros(2)

ax = plt.gca()
ax.margins(0)
ax.set(xlim=(posmin[0],posmax[0]), ylim=(posmin[1],posmax[1]))
plt.show()

