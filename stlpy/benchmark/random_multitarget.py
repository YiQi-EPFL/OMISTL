import numpy as np
import matplotlib.pyplot as plt

from .base import BenchmarkScenario
from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
from ..systems import DoubleIntegrator
from MPC.utils import random_obs


class RandomMultitarget(BenchmarkScenario):
    r"""
    A 2D mobile robot with double integrator dynamics must 
    navigate through a field of obstacles (grey, :math:`\mathcal{O}_i`)
    and reach at least one target of each color (:math:`\mathcal{T}_i^j`):

    .. math::

        \varphi = 
            \bigwedge_{i=1}^{N_c} \left( \bigvee_{j=1}^{N_t} F_{[0,T]} T_{i}^{j} \right) 
            \land G_{[0,T]} (\bigwedge_{k=1}^{N_o} \lnot O_k),

    :param num_obstacles:       number of obstacles, :math:`N_o`
    :param num_groups:          number of target groups/colors, :math:`N_c`
    :param targets_per_group:   number of targets in each group, :math:`N_t`
    :param T:                   time horizon of the specification
    :param seed:                (optional) seed for random generation of obstacle 
                                and target locations. Default is ``None``.
    """
    def __init__(self, num_obstacles, num_groups, targets_per_group, T, posmin, posmax, border_size, box_buffer,
              min_box_size, max_box_size, seed=None):
        self.T = T
        self.targets_per_group = targets_per_group
        self.spec_name = 'RandomMultitarget'

        # Set the seed for the random number generator (for reproducability)
        np.random.seed(seed)
        # Create the (randomly generated) set of rectangles, including obstacles and targets
        rec = random_obs(num_obstacles + num_groups*targets_per_group, posmin, posmax, border_size, box_buffer,
              min_box_size, max_box_size)
        obs = rec[:num_obstacles]
        obs = np.array(obs).reshape(num_obstacles, 4)
        self.obstacles = obs
        tar = rec[num_obstacles:]
        a = np.array(tar).reshape(num_groups,targets_per_group,4)
        self.targets = a
        self.T = T

    def random_obs(self, n_obs, posmin, posmax, border_size, box_buffer, min_box_size, max_box_size, max_iter=100):
        """ Generate random list of obstacles in workspace """
        obstacles = []
        itr = 0
        while itr < max_iter and len(obstacles) is not n_obs:
            xmin = (posmax[0] - border_size - max_box_size) * np.random.rand() + border_size
            xmax = xmin + min_box_size + (max_box_size - min_box_size) * np.random.rand()
            ymin = (posmax[1] - border_size - max_box_size) * np.random.rand() + border_size
            ymax = ymin + min_box_size + (max_box_size - min_box_size) * np.random.rand()
            obstacle = np.array([xmin - box_buffer, xmax + box_buffer, \
                                 ymin - box_buffer, ymax + box_buffer])
            intersecting = False
            for obs_2 in obstacles:
                intersecting = self.obs_intersect(obstacle, obs_2)
                if intersecting:
                    break
            if not intersecting:
                obstacles.append(obstacle)
            itr += 1

        if len(obstacles) is not n_obs:
            obstacles = []
        return obstacles

    def obs_intersect(self, obs_1, obs_2):
        intersect = True
        if obs_1[1] < obs_2[0] or \
                obs_2[1] < obs_1[0] or \
                obs_1[3] < obs_2[2] or \
                obs_2[3] < obs_1[2]:
            intersect = False
        return intersect

    def GetSpecification(self):
        # Specify that we must avoid all obstacles
        obstacle_formulas = []
        for obs in self.obstacles:
            obstacle_formulas.append(outside_rectangle_formula(obs, 0, 1, 6))
        obstacle_avoidance = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]

        # Specify that for each target group, we need to visit at least one
        # of the targets in that group
        target_group_formulas = []
        for target_group in self.targets:
            group_formulas = []
            for target in target_group:
                group_formulas.append(inside_rectangle_formula(target, 0, 1, 6))
            reach_target_group = group_formulas[0]
            for i in range(1, self.targets_per_group):
                reach_target_group = reach_target_group | group_formulas[i]
            target_group_formulas.append(reach_target_group)

        # Put all of the constraints together in one specification
        specification = obstacle_avoidance.always(0, self.T)
        for reach_target_group in target_group_formulas:
            specification = specification & reach_target_group.eventually(0, self.T)

        return specification

    def GetSystem(self):
        return DoubleIntegrator(2)

    def add_to_plot(self, ax):
        # Add red rectangles for the obstacles
        for obstacle in self.obstacles:
            ax.add_patch(make_rectangle_patch(*obstacle, color='k', alpha=0.5, zorder=-1))

        # Use the color cycle to choose the colors of each target group
        # (note that this won't work for more than 10 target groups)
        colors = plt.cm.tab10.colors
        for i, target_group in enumerate(self.targets):
            color = colors[i]
            for target in target_group:
                ax.add_patch(make_rectangle_patch(*target, color='b', alpha=0.7, zorder=-1))

        # set the field of view
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))
        ax.set_aspect('equal')
