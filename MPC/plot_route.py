import sys
sys.path.append("..")
import pickle, os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1, font="serif", style="white")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'#不设这个解不出#


def plot(optvals,prob_params,N,n_obs):
    Xopt = optvals[0]

    dataset_name = 'MPC_horizon_{}_obs_{}'.format(N, n_obs)

    relative_path = os.getcwd()
    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')
    outfile = open(config_fn, "rb")
    config = pickle.load(outfile)
    posmin = config[9]
    posmax = config[10]

    obstacles = []
    for ii_obs in range(n_obs):
        obs = prob_params['obstacles'][:, ii_obs]
        obstacles.append(obs)

    if len(obstacles) is n_obs:
        plt.axes()
        for obstacle in obstacles:
            rectangle = plt.Rectangle((obstacle[0], obstacle[2]),
            obstacle[1] - obstacle[0], obstacle[3] - obstacle[2],
                                      fc='white', ec='black')
            plt.gca().add_patch(rectangle)
            plt.axis('scaled')

        xg = prob_params['xg']
        x0 = prob_params['x0']
        circle = plt.Circle((x0[0], x0[1]), 0.04, fc='blue', ec="blue")
        plt.gca().add_patch(circle)

        # blue line is network prediction
        plt.plot(xg[0], xg[1], 'sr')
        # plt.quiver(Xopt[0,:], Xopt[1,:], Xopt[2,:], Xopt[3,:])#plot using arrows
        for jj in range(N):
            circle = plt.Circle((Xopt[0, jj], Xopt[1, jj]), 0.02, fc='red', ec="red")
            plt.gca().add_patch(circle)

        ax = plt.gca()
        ax.margins(0)
        ax.set(xlim=(posmin[0], posmax[0]), ylim=(posmin[1], posmax[1]))
        plt.show()


def plot_MPC(Xopt,ref_list,prob_params,N,n_obs):
    # plot results
    obstacles = []
    x0 = Xopt[0]
    dataset_name = 'MPC_horizon_{}_obs_{}'.format(N, n_obs)

    relative_path = os.getcwd()
    config_fn = os.path.join(relative_path, 'config', dataset_name + '.p')
    outfile = open(config_fn, "rb")
    config = pickle.load(outfile)
    posmin = config[9]
    posmax = config[10]


    for ii_obs in range(n_obs):
        # obs = test_data[0]['obstacles'][idx][:,ii_obs]
        obs = prob_params['obstacles'][:, ii_obs]
        obstacles.append(obs)

    if len(obstacles) is n_obs:
        plt.axes()
        for obstacle in obstacles:
            rectangle = plt.Rectangle((obstacle[0], obstacle[2]),
                                      obstacle[1] - obstacle[0], obstacle[3] - obstacle[2],
                                      fc='white', ec='black')
            plt.gca().add_patch(rectangle)
            plt.axis('scaled')

        for i in range(len(ref_list)):
            plt.plot(ref_list[i][0], ref_list[i][1], 'sr')

        for jj in range(len(Xopt[:, 0])):
            circle = plt.Circle((Xopt[jj][0], Xopt[jj][1]), 0.02, fc='black', ec="black")
            plt.gca().add_patch(circle)

    circle = plt.Circle((x0[0], x0[1]), 0.02, fc='red', ec="red")
    plt.gca().add_patch(circle)

    ax = plt.gca()
    ax.margins(0)
    ax.set(xlim=(posmin[0], posmax[0]), ylim=(posmin[1], posmax[1]))
    plt.show()

