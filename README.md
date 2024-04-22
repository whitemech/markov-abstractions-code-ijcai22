# Implementation of Markov Abstractions

This repository holds the implementation for "Markov Abstractions for PAC Reinforcement Learning in Non-Markov Decision Processes".

## Requirements

We use [Poetry](https://python-poetry.org/) to manage the virtual environment.
Create the virtual environment by running:

```
poetry install
```

Activate the environment shell:
```
poetry shell
```

## Training and evaluation

Once the virtual environment is activated, you may run experiments using:
```
python run.py --params-file <params-file>
```

Where `<params-file>` is a JSON file specifying the parameters 
necessary for running an experiment. Examples are provided in `experiments/definitions`.
An experiment log with the results will be saved in `experiments/logs` once the execution is finished.

Experiments can also be run by specifying the parameters via CLI. 
For a detailed explanation on the parameters used for running an experiment, please check `python run.py --help`:

```
  --params-file PARAMS_FILE
                        Params file with all the arguments.
  --alg-name 
                        Algorithm to use.
  --alg-params ALG_PARAMS
                        The algorithm parameters (formatted in JSON).
  --env-name 
                        The environment name.
  --env-params ENV_PARAMS
                        The environment parameters (formatted in JSON).
  --evaluation EVALUATION
                        Number of episodes between each evaluation.
  --seed SEED           
                        The random seed.
  --log-path LOG_PATH   
                        The logs output directory.
  --log-frequency LOG_FREQUENCY
                        Number of episodes in which logs are saved to disk.
```

Note that **training** and **evaluation** are done interchangeably when running an experiment.
While training, the agent's policy is evaluated at every `evaluation-frequency` number of episodes. 
We provide a script `experiments/plot.py` to plot the agent's performance given the logs of an experiment.

## Reproducibility

The experiments reported in the paper are reproducible. 
Please find in `experiments/definitions` the exact parameters used in our experiments. 
Each experiment was run 5 times and each file corresponds to one run, named by the run number. 
The files contain the exact random seed used for the results reported in the paper. 
You may reproduce them by following the running instructions above and using the provided parameters files. 
