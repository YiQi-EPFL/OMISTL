from abc import ABC, abstractmethod
from .base import BenchmarkScenario
from .common import (inside_rectangle_formula,
                     outside_rectangle_formula,
                     make_rectangle_patch)
from ..systems import DoubleIntegrator
import numpy as np

class Obstacles_Avoid(BenchmarkScenario):
    r"""
    A 2D mobile robot with double integrator dynamics must
    avoid an obstacle (:math:`\mathcal{O}`) before reaching a goal (:math:`\mathcal{G}`):

    .. math::

        \varphi = G_{[0,T]} \lnot \mathcal{O} \land F_{[0,T]} \mathcal{G}

    :param goal_bounds:      a tuple ``(xmin, xmax, ymin, ymax)`` defining a
                             rectangular goal region.
    :param obstacle_bounds:  a tuple ``(xmin, xmax, ymin, ymax)`` defining a
                             rectangular obstacle.
    :param T:                the time horizon for this scenario.
    """
    def __init__(self, T):
        self.T = T

    def GetSpecification(self):
        # Goal Reaching
        self.obstacles = []

        self.obstacles.append((1.25, 2.00, 1.20, 1.50))
        self.obstacles.append((1.25, 1.75, 0.20, 1.00))
        self.obstacles.append((0.30, 0.80, 1.50, 2.00))
        self.obstacles.append((2.50, 3.25, 1.60, 2.00))

        # for i in range(num_obstacles):
        #     x = np.random.uniform(0, 9)  # keep within workspace
        #     y = np.random.uniform(0, 9)
        #     self.obstacles.append((x, x + 2, y, y + 2))

        obstacle_formulas = []
        for obs in self.obstacles:
            obstacle_formulas.append(outside_rectangle_formula(obs, 0, 1, 6))
        obstacle_avoidance = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            obstacle_avoidance = obstacle_avoidance & obstacle_formulas[i]

        # Put all of the constraints together in one specification
        spec = obstacle_avoidance.always(0, self.T)

        return spec