import importlib
import pathlib
import json
import pickle
import datetime

from typing import Optional, Union, Dict, Any


def initialise_log(logs_path, params):
    pathlib.Path(logs_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{logs_path}/abstraction/hypotheses/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{logs_path}/rmax/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{logs_path}/dqn/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{logs_path}/random/").mkdir(parents=True, exist_ok=True)
    # Write arguments along with date/time of execution
    params["datetime"] = str(datetime.datetime.now())
    with open(f"{logs_path}/params.json", "w+") as f:
        json.dump(params, f, indent=2)
    del params["datetime"]

def update_log(logs_path, log_file, log):
    if log_file in ['q', 'count', 'reward', 'pdfa_learner_params']:
        with open(f"{logs_path}/{log_file}.pkl", "wb+") as f:
            pickle.dump(log, f)
    elif log_file == 'elapsed_time':
        with open(f"{logs_path}/params.json", "r+") as f:
            params = json.load(f)
        params["elapsed_time"] = log
        with open(f"{logs_path}/params.json", "w+") as f:
            json.dump(params, f, indent=2)
    else:
        with open(f"{logs_path}/{log_file}.json", 'w+') as f:
            json.dump(log, f, indent=2)

def parse_params_file(params_file):
    with open(params_file, 'r') as f:
        data = json.load(f)
    params = dict()
    params['alg_name'] = data['algorithm']['name']
    params['alg_params'] = data['algorithm']['params']
    params['env_name'] = data['environment']['name']
    params['env_params'] = data['environment']['params']
    params['evaluation'] = data['evaluation']
    params['log_path'] = data['log-path']
    params['log_frequency'] = data['log-frequency']
    params['seed'] = data['seed']
    return params

def get_env_spec(env_name, env_params):
    """Get the env spec from its name."""
    module = importlib.import_module(f"markov_abstractions.envs.specs.{env_name}")
    env_object = module.instantiate_env(env_params)
    return env_object

def get_sampling_policy(policy_name):
    """Get the sampling policy from its name."""
    module = importlib.import_module(f"markov_abstractions.policies")
    policy_class = getattr(module, policy_name)
    return policy_class
