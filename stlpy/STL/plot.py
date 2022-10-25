import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def reach_avoid_plot(optvals,scenario,prob_params):
    Xopt = optvals[0]
    spec = scenario.GetSpecification()
    spec.simplify()

    spec_name = scenario.spec_name
    N = scenario.T+1

    dataset_name = '{}_horizon_{}'.format(spec_name, N)

    # 加载设置文件地址，config = [dataset_name, prob_params, sampled_params]#
    relative_path = os.getcwd()
    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')

    config_file = open(config_fn, 'rb')
    config = pickle.load(config_file)
    config_file.close()  # 读取设置#

    dataset_name, _ , sampled_params, n_obs, num_probs, border_size, box_buffer, min_box_size, max_box_size, \
    posmin, posmax, velmin, velmax, n, m, obstacles = config


    goal_bounds = list(scenario.goal_bounds)    # (xmin, xmax, ymin, ymax)
    obstacle_bounds = list(scenario.obstacle_bounds)

    goal = plt.Rectangle((goal_bounds[0], goal_bounds[2]), \
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

    for jj in range(scenario.T):
        circle = plt.Circle((Xopt[0,jj],Xopt[1,jj]), 0.04, fc='black',ec="black")
        plt.gca().add_patch(circle)


    ax = plt.gca()
    ax.margins(0)
    ax.set(xlim=(posmin[0],posmax[0]), ylim=(posmin[1],posmax[1]))

    figure = ax.get_figure()
    return figure


def multi_targets_plot(optvals,scenario,prob_params):

    Xopt = optvals[0]
    spec = scenario.GetSpecification()
    spec.simplify()

    spec_name = scenario.spec_name
    N = scenario.T+1

    dataset_name = '{}_horizon_{}'.format(spec_name, N)

    colors = ['green', 'yellow', 'blue', 'grey', 'orange']
    for i in range(len(scenario.targets)):
        color = colors[i]
        for j in range(scenario.targets_per_group):
            target_bounds = list(scenario.targets[i][j])
            targets_bounds = plt.Rectangle((target_bounds[0], target_bounds[2]), \
                                           target_bounds[1] - target_bounds[0], target_bounds[3] - target_bounds[2], \
                                           fc=color, ec='black')
            plt.gca().add_patch(targets_bounds)
            plt.axis('scaled')

    for i in range(len(scenario.obstacles)):
        obstacle_bounds = list(scenario.obstacles[i])
        obstacle = plt.Rectangle((obstacle_bounds[0], obstacle_bounds[2]), \
                                 obstacle_bounds[1] - obstacle_bounds[0], obstacle_bounds[3] - obstacle_bounds[2], \
                                 fc='white', ec='black')
        plt.gca().add_patch(obstacle)


    x0 = prob_params['x0']
    circle = plt.Circle((x0[0],x0[1]), 0.04, fc='red',ec="red")
    plt.gca().add_patch(circle)

    for jj in range(scenario.T):
        circle = plt.Circle((Xopt[0,jj],Xopt[1,jj]), 0.04, fc='black',ec="black")
        plt.gca().add_patch(circle)

    posmin = np.zeros(2)
    posmax = np.array([10., 10., 1., 1.])

    ax = plt.gca()
    ax.margins(0)
    ax.set(xlim=(posmin[0],posmax[0]), ylim=(posmin[1],posmax[1]))
    figure = ax.get_figure()
    return figure
