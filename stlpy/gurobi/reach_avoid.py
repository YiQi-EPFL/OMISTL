#!/usr/bin/env python

##
#
# Set up, solve, and plot the solution for a simple
# reach-avoid problem, where the robot must avoid
# a rectangular obstacle before reaching a rectangular
# goal.
#
##
import random

import numpy as np
import matplotlib.pyplot as plt

from stlpy.benchmark import ReachAvoid
from stlpy.gurobi.guroby_micp import *
from stlpy.gurobi.drake_sos1 import *

# Specification Parameters
goal_bounds = (7,8,5,6)     # (xmin, xmax, ymin, ymax)
# goal_bounds = [2.90, 3.25, 2.00, 2.25]

obstacle_bounds = (3 ,5,4,6)
# obstacle_bounds = (1.25, 1.75, 0.20, 1.00)
T = 10


# Define the system and specification
scenario = ReachAvoid(goal_bounds, obstacle_bounds, T)
spec = scenario.GetSpecification()
sys = scenario.GetSystem()

# Specify any additional running cost (this helps the numerics in
# a gradient-based method)
Q = 1e-1*np.diag([0,0,1,1])   # just penalize high velocities
R = 1e-1*np.eye(2)

# Initial state
x_1 = np.random.randint(1,4)
x_2 = np.random.randint(2,7.5)


# x0 = np.array([x_1,x_2,0,0])
x0 = np.array([1.5,0.1,0,0])


# Choose a solver
# solver = GurobiMICPSolver(spec, sys, x0, T, robustness_cost=True)
#solver = DrakeMICPSolver(spec, sys, x0, T, robustness_cost=True)
solver = Sos1Solver(spec, sys, x0, T, robustness_cost=True)

# Set bounds on state and control variables
u_min = np.array([-0.5,-0.5])
u_max = np.array([0.5, 0.5])
x_min = np.array([0.0, 0.0, -1.0, -1.0])
x_max = np.array([10.0, 10.0, 1.0, 1.0])
solver.AddControlBounds(u_min, u_max)
#solver.AddStateBounds(x_min, x_max)

# Add quadratic running cost (optional)
solver.AddQuadraticCost(Q,R)

# Solve the optimization problem
x, u, _, _ = solver.Solve()
# print('loading network at ' + 'D:\Curious\CoCo-master\\MPC\solver_config\default_horizon_7.p')
#
# print('find best sulotion! ')
# print('solver time: ' + str(np.random.random()))


if x is not None:
    # Plot the
    ax = plt.gca()
    scenario.add_to_plot(ax)
    plt.scatter(*x[:2,:])
    plt.show()
