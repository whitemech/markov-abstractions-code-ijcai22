from matplotlib import pyplot as plt
import os, json
import numpy as np


font_sizes = {'font.size': 14, 'legend.fontsize': 14}
plt.rcParams.update(font_sizes)


def load_experiment_runs(experiment):
    experiment_path = f'experiments/logs/{experiment}'
    experiment_runs = list()
    for run in os.listdir(experiment_path):
        if run == '4': continue
        traces_log_path = f'{experiment_path}/{run}/logs/abstraction/trace_lengths.json'
        if os.path.exists(traces_log_path):
            with open(traces_log_path, 'r+') as f:
                print(f'- run {run}')
                traces_run = json.load(f)
            experiment_runs.append(traces_run)
            print(len(traces_run['trace_lengths']))
    return experiment_runs

def plot_trace_lengths(experiment_runs, interval=10000, episodes_cut=None, x_unit=100000):
    # Cut episodes according to the execution that has the minimum
    if episodes_cut is None:
        episodes_cut = min([len(run['trace_lengths']) for run in experiment_runs])
    print(f'- episodes cut: {episodes_cut}')

    # Get trace length from all runs
    trace_lengths = np.array([np.array(run['trace_lengths'][:episodes_cut]) for run in experiment_runs])
    safe_trace_lengths = np.array([np.array(run['safe_trace_lengths'][:episodes_cut]) for run in experiment_runs])
    candidate_trace_lengths = np.subtract(trace_lengths, safe_trace_lengths)
    avg_safe_trace_lengths = np.mean(safe_trace_lengths, axis=0)
    std_safe_trace_lengths = np.std(safe_trace_lengths, axis=0)
    avg_candidate_trace_lengths = np.mean(candidate_trace_lengths, axis=0)
    std_candidate_trace_lengths = np.std(candidate_trace_lengths, axis=0)
    # Plot
    avg_safe_trace_lengths = avg_safe_trace_lengths[::interval]
    std_safe_trace_lengths = std_safe_trace_lengths[::interval]
    avg_candidate_trace_lengths = avg_candidate_trace_lengths[::interval]
    std_candidate_trace_lengths = std_candidate_trace_lengths[::interval]
    x_axis = (np.array(list(range(len(avg_candidate_trace_lengths))))*interval)/x_unit
    plt.plot(x_axis, avg_candidate_trace_lengths, label='Non-safe states', color='black') 
    plt.fill_between(x_axis, avg_candidate_trace_lengths + std_candidate_trace_lengths, avg_candidate_trace_lengths - std_candidate_trace_lengths, alpha=0.07, color='black') 
    plt.plot(x_axis, avg_safe_trace_lengths, label='Safe states', color='green') 
    plt.fill_between(x_axis, avg_safe_trace_lengths + std_safe_trace_lengths, avg_safe_trace_lengths - std_safe_trace_lengths, alpha=0.07, color='green') 



def main():
    # Cut the x-axis at a given point
    # (no need to show all evaluations after it converged)
    episodes_cut = 1000000
    interval = 10000
    x_unit = 100000

    # Experiments to plot (the folder name of the experiment)
    experiments = [
        'grid_rmax_abstraction',
    ]

    # Iterate experiments and plot
    for experiment in experiments:
        print(experiment)
        experiment_runs = load_experiment_runs(experiment)
        plot_trace_lengths(experiment_runs, interval=interval, episodes_cut=episodes_cut)

    # Set labels, grid, legend, and save figure
    plt.tight_layout(pad=1.5) 
    # plt.xscale('log',base=10) 
    plt.ylabel('Episode length') 
    plt.xlabel(f'Episodes (unit=10$^{int(np.log10(x_unit))}$)') 
    plt.grid() 
    plt.legend(loc='upper right', borderpad=0.3, labelspacing=0.4) 
    plt.savefig(f'plot.pdf') 
    plt.close() 



if __name__ == '__main__':
    main()
