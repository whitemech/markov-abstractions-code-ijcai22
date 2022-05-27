import time
import argparse
import json
import cProfile
import pstats
import io
import copy
from math import ceil, log

from markov_abstractions.abstraction_module.core import AbstractionModule
from markov_abstractions.agents.rmax import RMaxAgent
from markov_abstractions.agents.random import RandomAgent
from markov_abstractions.envs.rdp_env import RDPEnv
from markov_abstractions.utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        "run",
        description="Run an experiment with Markov Abstractions.",
    )
    parser.add_argument("--params-file", type=str, help="Params file with all the arguments.")
    parser.add_argument("--alg-name", type=str, default="rmax", choices=["rmax", "dqn", "random"], help="Algorithm to use.")
    parser.add_argument("--alg-params", type=json.loads, default='{ \
                            "abstraction": true, \
                            "combine_obs_state": false, \
                            "string_propagation": true, \
                            "alpha": 2, \
                            "alpha_0": 128, \
                            "delay": 1, \
                            "delta": 0.2, \
                            "epsilon": 0.2, \
                            "mu": 0.224, \
                            "n": 50, \
                            "nb_episodes": 500001, \
                            "sampling_policy": "RandomPolicy", \
                            "termination_schedule": 1000, \
                            "use_vmax": true, \
                            "m_0": 10000, \
                            "net_arch": [516, 256, 64], \
                            "buffer_size": 1000, \
                            "batch_size": 128, \
                            "learning_rate": 1e-3, \
                            "learning_starts": 5000, \
                            "target_update_interval": 500, \
                            "train_freq": 100, \
                            "exploration_final_eps": 0.01, \
                            "exploration_fraction": 0.1, \
                            "total_timesteps": 10000001, \
                            "log_interval": 100, \
                            "verbose": 1 \
                        }', help="The algorithm parameters (formatted in JSON).")
    parser.add_argument("--env-name", default="grid", 
                        choices=["rotating_maze", "rotating_mab", "malfunction_mab", 
                        "cheat_mab", "driving_agent", "enemy_corridor", "keys_world", 
                        "cookie_world", "symbol_world", "grid"], help="The environment name.")
    parser.add_argument("--env-params", type=json.loads, default='{ \
                            "specification": {}, \
                            "markovian": false, \
                            "use_stop_prob": true, \
                            "flicker_prob": 0.2, \
                            "nb_steps": 10 \
                        }', help="The environment parameters (formatted in JSON).")
    parser.add_argument("--evaluation", type=json.loads, default='{ \
                            "episodes": 50, \
                            "frequency": 10000, \
                            "steps": 10 \
                        }', help="Number of episodes between each evaluation.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--log-path", default=f"experiments/logs/{time.time()}/0/logs", 
                        type=str, help="The logs output directory.")
    parser.add_argument("--log-frequency", default=10000, type=int,
                        help="Number of episodes in which logs are saved to disk.")
    return parser.parse_args()


def main(
    alg_name: str,
    alg_params: dict,
    env_name: str,
    env_params: dict,
    evaluation: dict,
    log_path: str,
    log_frequency: int,
    seed: int
):
    # Enable profiling
    pr = cProfile.Profile()
    pr.enable()

    # Get environment specification
    env_spec = get_env_spec(env_name, env_params['specification'])
    # Generate RDP environment
    env = RDPEnv(env_spec, markovian=env_params['markovian'],
                    use_stop_prob=env_params['use_stop_prob'], episode_length=env_params['nb_steps'], flicker_prob=env_params['flicker_prob'], rand_seed=seed)

    # If use abstraction
    if alg_params['abstraction']:
        # Compute current delay
        if alg_params['delay'] == 0:
            alg_params['delay'] = ceil(log((1/alg_params['mu']), len(env.actions)))
        # Get the policy for sampling remaining of episodes
        policy_class = get_sampling_policy(alg_params['sampling_policy'])
        if alg_name == "rmax":
            # Instantiate agent and abstraction
            env = AbstractionModule(
                env,
                mu=alg_params['mu'],
                n=alg_params['n'],
                epsilon=alg_params['epsilon']/2,
                delta=alg_params['delta']/2,
                alpha=alg_params['alpha'],
                alpha_0=alg_params['alpha_0'],
                d=alg_params['delay'],
                policy=policy_class,
                rand_seed=seed,
                termination_schedule=alg_params['termination_schedule'],
                string_propagation=alg_params['string_propagation'],
                combine_obs_state=alg_params['combine_obs_state'],
                log_frequency=log_frequency,
                log_path=log_path
            )
            agent = RMaxAgent(
                env,
                stop_prob=env.stop_prob,
                epsilon=alg_params['epsilon']/2,
                delta=alg_params['delta']/2,
                m_0=alg_params['m_0'],
                rand_seed=seed,
                use_vmax=alg_params['use_vmax'],
                evaluation_frequency=evaluation['frequency'],
                evaluation_episodes=evaluation['episodes'],
                evaluation_steps=evaluation['steps'],
                log_frequency=log_frequency,
                log_path=log_path
            )
            # Train
            train_start_time = time.time()
            agent.train_on_abstraction(alg_params['nb_episodes'])
            train_time = time.time() - train_start_time
            # Update logs
            update_log(log_path, 'elapsed_time', train_time)
        elif alg_name == "random":
            # Instantiate agent and abstraction
            env = AbstractionModule(
                env,
                mu=alg_params['mu'],
                n=alg_params['n'],
                epsilon=alg_params['epsilon']/2,
                delta=alg_params['delta']/2,
                alpha=alg_params['alpha'],
                alpha_0=alg_params['alpha_0'],
                d=alg_params['delay'],
                policy=policy_class,
                rand_seed=seed,
                termination_schedule=alg_params['termination_schedule'],
                string_propagation=alg_params['string_propagation'],
                combine_obs_state=alg_params['combine_obs_state'],
                log_frequency=log_frequency,
                log_path=log_path
            )
            agent = RandomAgent(
                env,
                stop_prob=env.stop_prob,
                epsilon=alg_params['epsilon']/2,
                delta=alg_params['delta']/2,
                m_0=alg_params['m_0'],
                rand_seed=seed,
                use_vmax=alg_params['use_vmax'],
                evaluation_frequency=evaluation['frequency'],
                evaluation_episodes=evaluation['episodes'],
                evaluation_steps=evaluation['steps'],
                log_frequency=log_frequency,
                log_path=log_path
            )
            # Train
            train_start_time = time.time()
            agent.train_on_abstraction(alg_params['nb_episodes'])
            train_time = time.time() - train_start_time
            # Update logs
            update_log(log_path, 'elapsed_time', train_time)
    # Use RMax only
    else:
        # Instantiate agent and environment
        env = env
        agent = RMaxAgent(
            env,
            stop_prob=env.stop_prob,
            epsilon=alg_params['epsilon']/2,
            delta=alg_params['delta']/2,
            m_0=alg_params['m_0'],
            rand_seed=seed,
            use_vmax=alg_params['use_vmax'],
            evaluation_frequency=evaluation['frequency'],
            evaluation_episodes=evaluation['episodes'],
            evaluation_steps=evaluation['steps'],
            log_frequency=log_frequency,
            log_path=log_path
        )
        # Train
        train_start_time = time.time()
        agent.train(alg_params['nb_episodes'])
        train_time = time.time() - train_start_time
        # Update logs
        update_log(log_path, 'elapsed_time', train_time)

    # Disable profiling and save stats
    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open(f'{log_path}/profiler.txt', 'w+') as f:
        f.write(s.getvalue())


if __name__ == '__main__':
    arguments = parse_args()

    if arguments.params_file:
        params = parse_params_file(arguments.params_file)
        # Create output directory and initialise logs
        initialise_log(params["log_path"], params)
        # Run
        main(**params)
    else:
        arguments.__dict__.pop('params_file')
        # Create output directory and initialise logs
        initialise_log(arguments.__dict__["log_path"], arguments.__dict__)
        # Run
        main(**arguments.__dict__)
