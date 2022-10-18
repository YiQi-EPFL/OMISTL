## set MPC parameters
import numpy as np
from MPC_prob import MPC
from solvers.OMISTL import OMISTL
import cvxpy as cp
import pickle

def MPC_controller(x0,ref_list,prob,MPC_obj,T=200):

    #initial MPC problem
    T = T
    prob = prob
    system = 'free_flyer'
    prob_features = ['x0', 'xg', 'obstacles']

    # initial neural network and parameters
    x_init = x0
    prob_params = {}
    prob_params['x0'] = x0
    prob_params['xg'] = ref_list[0]
    ## get initial solution
    obstacles = prob.configs[-1]
    N = prob.configs[1][0]
    prob_params['obstacles'] = np.reshape(np.concatenate(obstacles, axis=0), (prob.n_obs,4)).T
    prob_success, cost, total_time, n_evals, optvals, y_guess = MPC_obj.forward(prob_params, solver=cp.GUROBI, max_evals=10)

    i=0
    t_idx=0
    ##
    Xopt = np.vstack((x0, optvals[0][:,t_idx+1]  ))
    Uopt = optvals[1][:,t_idx]
    Yopt = optvals[2][:,t_idx]
    OPT = []
    T_total = 0

    ## start recursive computation for MPC
    n=0
    while i<T:
        x = optvals[0][:,t_idx+1]
        optvals_origin = optvals
        prob_params['x0'] = x
        prob_success = False
        try:
            prob_success, cost, total_time, n_evals, optvals, y_guess = MPC_obj.forward(prob_params, solver=cp.GUROBI, max_evals=10)
        except (KeyboardInterrupt, SystemExit):
                raise
        except:
            print('solver failed')
            prob_success = False
        ## if predicted solution feasible, update the next position as new parameter,
        ## store the solution into Xopt,Uopt,Yopt, and also store the solution as optvals_origin, in case of next position is infeasible
        if prob_success:
            t_idx=0
            print('Found solution at '+ 't =  ' +str(i) + ". n_evals = " + str(n_evals) )
            Xopt = np.vstack((Xopt, optvals[0][:,t_idx+1]  ))
            Uopt = np.vstack((Uopt, optvals[1][:,t_idx]  ))
            Yopt = np.vstack((Yopt, optvals[2][:,t_idx]  ))
            optvals_origin = optvals
            T_total += total_time
            i+=1
            #only when close to the reference point, change the destination as next point
            if sum(np.absolute((Xopt[-2,:] - ref_list[n])))<0.05:
                n+=1
                #after exploring all the reference points, return to x0
                if n >= len(ref_list):
                    prob_params['xg'] = x_init
                    n -=1
                else:
                    prob_params['xg'] = ref_list[n]
        ## if predicted solution unfeasible, continue original path using the previous solution
        else:
            t_idx+=1
            print('continue original path at t= '+ str(i))
            i+=1
            optvals = optvals_origin
            # we have N-1 available positions from the original solutions, if they are all infeasible, try to set the
            # velocity as 0 and then solve the MICP problem rather than ML to provide a feasible solution to continue
            if t_idx >=N-1:
                x = optvals[0][:,t_idx]
                x[2:] = 0
                prob_params['x0'] = x
                prob_success = False
                try:
                    prob_success, _, solvetime_g, optvals = prob.solve_micp(prob_params, solver=cp.GUROBI)
                except:
                    print('Gurobi failed at '+ 't =  ' +str(i) + ".")
                if prob_success:
                    print('Found solution at '+ 't =  ' +str(i) + ". n_evals = " + str(n_evals) )
                    t_idx=0
                    Xopt = np.vstack((Xopt, optvals[0][:,t_idx+1]  ))
                    Uopt = np.vstack((Uopt, optvals[1][:,t_idx]  ))
                    Yopt = np.vstack((Yopt, optvals[2][:,t_idx]  ))
                    optvals_origin = optvals
                    T_total += solvetime_g
                    i+=1
                    continue
                ## if prediction infeasible and all points of the previous solution have been used up, break and return solver failed
                else:
                    print('Gurobi failed at '+ 't =  ' +str(i) + ".")
                    break
            Xopt = np.vstack((Xopt, optvals[0][:,t_idx+1]))
            Uopt = np.vstack((Uopt, optvals[1][:,t_idx]))
            Yopt = np.vstack((Yopt, optvals[2][:,t_idx]))

    print('total solving time for {} steps is '.format(T) + str(T_total))
    print('avarage solving time at each step is ' + str(T_total/T))
    return Xopt, Uopt, Yopt, T_total