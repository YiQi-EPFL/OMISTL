from stlpy.STL import LinearPredicate
import numpy as np
from guroby_micp import GurobiMICPSolver
from gurobipy import GRB
import math
from gray import GrayCode

class Sos1Solver(GurobiMICPSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`,
    solve the optimization problem

    .. math::

        \min & -\\rho^{\\varphi}(y_0,y_1,\dots,y_T) + \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = A x_t + B u_t

        & y_{t} = C x_t + D u_t

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    using mixed-integer convex programming. This method uses fewer binary variables
    by encoding disjunction with a Special Ordered Set of Type 1 (SOS1) constraint.

    .. note::

        This class implements the encoding described in

        Kurtz V, et al.
        *Mixed-Integer Programming for Signal Temporal Logic with Fewer Binary
        Variables*. IEEE Control Systems Letters, 2022. https://arxiv.org/abs/2204.06367.


    .. warning::

        Drake must be compiled from source to support the Gurobi MICP solver.
        See `<https://drake.mit.edu/from_source.html>`_ for more details.

        Drake's naive branch-and-bound solver does not require Gurobi or Mosek, and
        can be used with the ``bnb`` solver option, but this tends to be very slow. 

    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    :param solver:          (optional) String describing the solver to use. Must be one
                            of 'gurobi', 'mosek', or 'bnb'.
    :param presolve:        (optional) A boolean indicating whether to use gurobi's
                            presolve routines. Only affects the gurobi solver. Default is ``True``.
    :param verbose:         (optional) A boolean indicating whether to print detailed
                            solver info. Default is ``True``.
    """
    def __init__(self, spec, sys, x0, T, M=1000, robustness_cost=True, 
            solver='gurobi', presolve=True, verbose=True):
        super().__init__(spec, sys, x0, T, M, robustness_cost=robustness_cost,
                presolve=presolve, verbose=verbose)

    def AddSOS1SubformulaConstraints(self, formula, z, t):
        """
        Given an STLFormula (formula) and a binary variable (z),
        add constraints to the optimization problem such that z
        takes value 1 only if the formula is satisfied (at time t).

        If the formula is a predicate, this constraint uses the "big-M"
        formulation

            A[x(t);u(t)] - b + (1-z)M >= 0,

        which enforces A[x;u] - b >= 0 if z=1, where (A,b) are the
        linear constraints associated with this predicate.

        If the formula is not a predicate, we recursively traverse the
        subformulas associated with this formula, adding new binary
        variables z_i for each subformula and constraining

            z <= z_i  for all i

        if the subformulas are combined with conjunction (i.e. all
        subformulas must hold), or otherwise constraining

            z <= sum(z_i)

        if the subformulas are combined with disjuction (at least one
        subformula must hold).
        """
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, LinearPredicate):
            # a.T*y - b + (1-z)*M >= rho
            self.model.addConstr(formula.a.T@self.y[:,t] - formula.b + (1-z)*self.M >= self.rho)

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            nz = len(formula.subformula_list)

            if formula.combination_type == "and":
                z_subs = self.model.addVars(nz,1, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
                self.model.addConstr(z <= z_subs)
            else:  # combination_type == "or":

                lambda_,binary = self.addsos1constraint(nz + 1)
                z_subs = lambda_[1:][np.newaxis].T
                self.model.addConstr(1-z == lambda_[0] )

            for i, subformula in enumerate(formula.subformula_list):
                t_sub = formula.timesteps[i]
                self.AddSOS1SubformulaConstraints(subformula, z_subs[i], t+t_sub)

    def addsos1constraint(self,nz):
        lambd = self.model.addVars(nz, lb=0.0, vtype=GRB.CONTINUOUS)
        num_y = math.ceil(math.log2(nz))
        a = GrayCode()
        X = a.getGray(num_y)
        Xlist = []
        for i in range(2 ** num_y):
            Xlist += X[i]
        Xarr = list(map(int, Xlist))
        num_row = int(len(Xarr) / num_y)
        Mat = np.array(Xarr).reshape(num_row, num_y)
        binary_encoding = Mat[:nz, :]
        y = self.model.addVars(num_y, vtype=GRB.BINARY)
        ##add sos1 constraints
        self.model.addConstr(sum(lambd) == 1)
        for j in range(0, num_y):
            lambda_sum1 = 0
            lambda_sum2 = 0
            for k in range(0, nz):
                if binary_encoding[k, j] == 1:
                    lambda_sum1 += lambd[k]
                elif binary_encoding[k, j] == 0:
                    lambda_sum2 += lambd[k]
                else:
                    print("Runtime_error: The binary_encoding entry can be only 0 or 1.")
                    break
            self.model.addConstr(lambda_sum1 <= y[j])
            self.model.addConstr(lambda_sum1 <= 1 - y[j])
        return lambd, y


