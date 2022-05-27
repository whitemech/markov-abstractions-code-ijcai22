import json
import os

import numpy as np
from matplotlib import pyplot as plt


font_sizes = {'font.size': 14, 'legend.fontsize': 13.2}
plt.rcParams.update(font_sizes)


def load_experiment_evaluation_runs(experiment):
    experiment_path = f'experiments/logs/{experiment}'
    experiment_evaluation_runs = list()
    for run in os.listdir(experiment_path):
        evaluation_log_path = f'{experiment_path}/{run}/logs/evaluations.json'
        if os.path.exists(evaluation_log_path):
            with open(evaluation_log_path, 'r+') as f:
                print(f'- run {run}')
                evaluation_run = json.load(f)
            experiment_evaluation_runs.append(evaluation_run)
    return experiment_evaluation_runs

def compute_evaluation_average_rewards_per_run(experiment_runs): 
    # Compute the average reward of each episode
    average_rewards_per_run = dict()
    for run_num, run_data in enumerate(experiment_runs):
        average_rewards_per_evaluation_episode = dict()
        for evaluation_episode, evaluation_data in run_data.items():
            # Get the total reward per episode and number of steps
            reward_per_episode = np.array(evaluation_data['episode_rewards'], dtype=float)
            steps_per_episode = np.array(evaluation_data['episode_steps'], dtype=int)
            # Divide the total reward by the number of steps per episode
            average_reward_per_episode = np.divide(reward_per_episode, steps_per_episode)
            average_reward_per_episode = np.nan_to_num(average_reward_per_episode)  # replace nan by 0
            # Compute the mean of the 50 episodes 
            average_reward_per_episode = np.mean(average_reward_per_episode)
            average_rewards_per_evaluation_episode[evaluation_episode] = average_reward_per_episode
        average_rewards_per_run[run_num] = average_rewards_per_evaluation_episode
    return average_rewards_per_run

def plot_evaluation_average_rewards(avg_rewards_per_run, color, legend, x_cut, x_unit):
    # Get average rewards from all runs
    avg_rewards_all_runs = list()
    for run_num, run_data in avg_rewards_per_run.items():
        run_average_rewards = np.array(list(run_data.values()), dtype=object)
        avg_rewards_all_runs.append(run_average_rewards) 
    # Cut evaluations according to the execution that has the minimum
    eval_cut = min([len(avg_rewards) for avg_rewards in avg_rewards_all_runs])
    print(f'- total evaluations: {eval_cut}')
    for i, avg_rewards_run in enumerate(avg_rewards_all_runs):
        avg_rewards_all_runs[i] = avg_rewards_run[:eval_cut]
    avg_rewards_all_runs = np.array(avg_rewards_all_runs) 
    # Compute mean and stderr 
    mean = np.mean(avg_rewards_all_runs, axis=0, dtype=float)
    stderr = np.std(avg_rewards_all_runs, axis=0, dtype=float)
    # Evaluations
    evaluations = np.array([i*15000 for i in range(len(mean))])
    # Set line style, width, and marker
    ls = "-"
    lw = 1.5
    marker = 's'
    markersize = 5
    # NOTE For Cheat MAB, Rotating Maze, and Rotating MAB v2:
    # markevery = 20
    # NOTE For Flickering Grid:
    # markevery = 10
    # NOTE For the other domains:
    markevery = 3
    if 'random' in legend.lower(): 
        ls = "--"
        lw = 2
        marker = 'x'
        markersize = 8
    elif 'rmax' in legend.lower(): 
        ls = "-"
        marker = 'o'
        markersize = 4
    # Plot
    x_axis = evaluations[:x_cut] / x_unit
    y_axis = mean[:x_cut]
    stderr = stderr[:x_cut]
    plt.plot(x_axis, y_axis, label=legend, color=color, ls=ls, lw=lw, marker=marker, markersize=markersize, markevery=markevery) 
    plt.fill_between(x_axis, y_axis + stderr, y_axis - stderr, alpha=0.07, color=color) 
    


def main():
    # Cut the x-axis at a given point
    # (no need to show all evaluations after it converged)
    # x_unit = 100000
    x_unit = 10000 # Enemy corridor
    # x_cut = 50 # Rotating MAB
    # x_cut = 30 # Malfunction MAB
    # x_cut = 200 # Cheat MAB
    # x_cut = 103 # Rotating Maze
    # x_cut = 334 # Rotating MAB v2
    x_cut = 15 # Enemy Corridor
    # x_cut = 100 # Flickering Grid

    # Experiments to plot (the folder name of the experiment)
    experiments = [
        # Rotating MAB
        # 'rotating_mab_k_2_rmax_baseline',
        # 'rotating_mab_k_4_rmax_baseline',
        # 'rotating_mab_k_6_rmax_baseline',
        # 'rotating_mab_k_2_random_baseline',
        # 'rotating_mab_k_4_random_baseline',
        # 'rotating_mab_k_6_random_baseline',
        # 'rotating_mab_k_2_rmax_abstraction',
        # 'rotating_mab_k_4_rmax_abstraction',
        # 'rotating_mab_k_6_rmax_abstraction',

        # Malfunction MAB
        # 'malfunction_mab_k_3_rmax_baseline',
        # 'malfunction_mab_k_5_rmax_baseline',
        # 'malfunction_mab_k_3_random_baseline',
        # 'malfunction_mab_k_5_random_baseline',
        # 'malfunction_mab_k_3_rmax_abstraction',
        # 'malfunction_mab_k_5_rmax_abstraction',

        # Cheat MAB
        # 'cheat_mab_k_3_rmax_baseline',
        # 'cheat_mab_k_4_rmax_baseline',
        # 'cheat_mab_k_3_random_baseline',
        # 'cheat_mab_k_4_random_baseline',
        # 'cheat_mab_k_3_rmax_abstraction',
        # 'cheat_mab_k_4_rmax_abstraction',

        # Rotating Maze
        # 'rotating_maze_k_1_rmax_baseline',
        # 'rotating_maze_k_2_rmax_baseline',
        # 'rotating_maze_k_3_rmax_baseline',
        # 'rotating_maze_k_1_random_baseline',
        # 'rotating_maze_k_2_random_baseline',
        # 'rotating_maze_k_3_random_baseline',
        # 'rotating_maze_k_1_rmax_abstraction',
        # 'rotating_maze_k_2_rmax_abstraction',
        # 'rotating_maze_k_3_rmax_abstraction',

        # Rotating MAB v2
        # 'rotating_mab_v2_k_8_rmax_baseline',
        # 'rotating_mab_v2_k_8_random_baseline',
        # 'rotating_mab_v2_k_8_rmax_abstraction',
        

        # Enemy Corridor
        'enemy_corridor_k_8_rmax_baseline',
        'enemy_corridor_k_16_rmax_baseline',
        'enemy_corridor_k_32_rmax_baseline',
        'enemy_corridor_k_64_rmax_baseline',
        'enemy_corridor_k_8_random_baseline',
        'enemy_corridor_k_16_random_baseline',
        'enemy_corridor_k_32_random_baseline',
        'enemy_corridor_k_64_random_baseline',
        'enemy_corridor_k_8_rmax_abstraction',
        'enemy_corridor_k_16_rmax_abstraction',
        'enemy_corridor_k_32_rmax_abstraction',
        'enemy_corridor_k_64_rmax_abstraction',

        # Flickering Grid
        # 'grid_rmax_baseline',
        # 'grid_random_baseline',
        # 'grid_rmax_abstraction',
    ]
    # the colors for the experiments above, respectively
    colors = [
        # Rotating MAB
        # 'black',
        # 'dimgray',
        # 'silver',
        # 'darkred',
        # 'red',
        # 'salmon',
        # 'darkgreen',
        # 'limegreen',
        # 'mediumseagreen',

        # Malfunction MAB
        # 'black',
        # 'dimgray',
        # 'darkred',
        # 'red',
        # 'darkgreen',
        # 'limegreen',

        # Cheat MAB
        # 'black',
        # 'dimgray',
        # 'darkred',
        # 'red',
        # 'darkgreen',
        # 'limegreen',

        # Rotating Maze
        # 'black',
        # 'dimgray',
        # 'silver',
        # 'darkred',
        # 'red',
        # 'salmon',
        # 'darkgreen',
        # 'limegreen',
        # 'mediumseagreen',

        # Rotating MAB v2
        # 'black',
        # 'red',
        # 'limegreen',

        # Enemy Corridor
        'black',
        'darkgray',
        'dimgray',
        'silver',
        'darkred',
        'red',
        'salmon',
        'lightsalmon',
        'darkgreen',
        'limegreen',
        'mediumseagreen',
        'turquoise',

        # Flickering Grid
        # 'black',
        # 'red',
        # 'limegreen',
    ]
    # the legend names for the experiments above, respectively
    legends = [
        # Rotating MAB
        # 'k=2 RMax',
        # 'k=4 RMax',
        # 'k=6 RMax',
        # 'k=2 Random',
        # 'k=4 Random',
        # 'k=6 Random',
        # 'k=2 Abstraction',
        # 'k=4 Abstraction',
        # 'k=6 Abstraction',

        # Malfunction MAB
        # 'k=3 RMax',
        # 'k=5 RMax',
        # 'k=3 Random',
        # 'k=5 Random',
        # 'k=3 Abstraction',
        # 'k=5 Abstraction',

        # Cheat MAB
        # 'k=3 RMax',
        # 'k=4 RMax',
        # 'k=3 Random',
        # 'k=4 Random',
        # 'k=3 Abstraction',
        # 'k=4 Abstraction',

        # Rotating Maze
        # 'k=1 RMax',
        # 'k=2 RMax',
        # 'k=3 RMax',
        # 'k=1 Random',
        # 'k=2 Random',
        # 'k=3 Random',
        # 'k=1 Abstraction',
        # 'k=2 Abstraction',
        # 'k=3 Abstraction',

        # Rotating MAB v2
        # 'k=8 RMax',
        # 'k=8 Random',
        # 'k=8 Abstraction',

        # Enemy Corridor
        'k=8 RMax',
        'k=16 RMax',
        'k=32 RMax',
        'k=64 RMax',
        'k=8 Random',
        'k=16 Random',
        'k=32 Random',
        'k=64 Random',
        'k=8 Abstraction',
        'k=16 Abstraction',
        'k=32 Abstraction',
        'k=64 Abstraction',

        # Flickering Grid
        # 'RMax',
        # 'Random',
        # 'Abstraction',
    ]

    # Iterate experiments and plot
    for num, experiment in enumerate(experiments):
        print(experiment)
        experiment_evaluation_runs = load_experiment_evaluation_runs(experiment)
        evaluation_avg_rewards = compute_evaluation_average_rewards_per_run(experiment_evaluation_runs)
        plot_evaluation_average_rewards(evaluation_avg_rewards, colors[num], legend=legends[num], x_cut=x_cut, x_unit=x_unit)

    # Set labels, grid, legend, and save figure
    plt.tight_layout(pad=1.5) 
    # plt.xscale('log', base=10) 
    plt.ylabel('Average Reward') 
    plt.xlabel(f'Episodes (unit=10$^{int(np.log10(x_unit))}$)') 
    plt.grid() 
    plt.legend(loc='lower right', borderpad=0.3, labelspacing=0.4) 
    plt.savefig(f'plot.pdf', bbox_inches='tight') 
    plt.close() 



if __name__ == '__main__':
    main()
