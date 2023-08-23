from markov_abstractions.abstraction_module.stream_learner.params import StreamParams
from markov_abstractions.abstraction_module.stream_learner.params import StreamParams
from markov_abstractions.abstraction_module.stream_learner.core import StreamPDFALearner
from markov_abstractions.policies import RandomPolicy
from markov_abstractions.utils import update_log

from gym.spaces import Box, Discrete, MultiDiscrete
from itertools import product
from math import inf, sqrt, log
import copy
import numpy as np
import gym


class AbstractionModule(gym.Wrapper):

    def __init__(self,
                 env=None,
                 mu=2,
                 n=0.4,
                 epsilon=0.05,
                 delta=0.01,
                 alpha=1.1,
                 alpha_0=16,
                 d=1,
                 policy=None,
                 rand_seed=0,
                 termination_schedule=1000,
                 string_propagation=True,
                 combine_obs_state=True,
                 log_frequency=10000,
                 log_path='log'
                 ):
        # Initialise random number generator
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(rand_seed)

        # Policy for sampling remaining of episodes
        if policy == None:
            self.policy = RandomPolicy
        else:
            self.policy = policy

        # Initialise environment
        super().__init__(env)
        self.A = env.actions
        self.O = env.observations
        self.R = env.rewards

        # Initialise Stream PDFA Learner
        self.stream_params = StreamParams(
            nb_actions=len(self.A),
            nb_observations=len(self.O),
            nb_rewards=len(self.R),
            Gamma=self.A,
            Sigma=set(product(self.O, self.R)),
            mu=mu,
            n=n,
            epsilon=epsilon,
            delta=delta,
            alpha=alpha,
            alpha_0=alpha_0,
            d=d,
            string_propagation=string_propagation,
            log_path=log_path
        )
        self.stream_learner = StreamPDFALearner(self.stream_params)

        self.terminal_symbol = 'terminal'

        # Stats
        self.t = 0
        self.termination_schedule = termination_schedule
        self.termination_time = (0, '')
        self.trace_lengths = list()
        self.safe_trace_lengths = list()
        self.log_frequency = log_frequency
        self.log_path = log_path

        # Log params
        print("Params:", self.stream_params)
        update_log(f'{self.log_path}/abstraction', 'pdfa_learner_params', self.stream_learner.params.__dict__)

    def step(self, action, agent=False):
        done = False
        observation = None
        reward = 0
        info = dict()
        s_ho = self.s_h
        if self.s_h in self.stream_learner.graph.safe:
            # Perform action in the actual environment
            observation, reward, done, info = super().step(action)
            # Append to histories
            self.trace.append((action, (observation, reward)))
            # Compute next state s_ho
            next_symbol = (action, (observation, reward))
            s_ho = self.stream_learner.graph.transitions.get(self.s_h, {}).get(next_symbol)
        s_ho_masked = self.terminal_symbol if done else s_ho
        # Check if episode is over or s_h is not safe
        if done or s_ho not in self.stream_learner.graph.safe:
            # Observe the candidate state
            if agent and self.s_h in self.stream_learner.graph.safe:
                agent.observe(self.s_h, action, reward, s_ho_masked)
            # Stats
            self.safe_trace_lengths.append(len(self.trace))
            # If not done, sample until it is over
            if not done:
                done = self.sample_rest_of_episode()
            # Stats
            self.trace_lengths.append(len(self.trace))
            # Consume trace
            merges, hyp = self.stream_learner.consume(self.trace)
            # Perform merges
            if agent:
                agent.merge_states(merges)
            # Perform termination test
            # if self.t % self.termination_schedule == 0:
            #     agent_terminated = agent.termination_test()
            #     if agent_terminated:
            #         self.termination_time = (self.t, 'termination_test = True')
            # Save logs
            if self.t % self.log_frequency == 0:
                self._save_logs()
            self.t += 1
        self.s_h = s_ho

        return s_ho_masked, reward, done, info

    def evaluation_step(self, action):
        # Perform action in the actual environment

        observation, reward, done, info = self.eval_step(action)
        self.trace.append((action, (observation, reward)))

        # Compute next state s_ho
        next_symbol = (action, (observation, reward))
        s_ho = self.stream_learner.graph.transitions.get(self.s_h, {}).get(next_symbol)
        s_ho_masked = self.terminal_symbol if done else s_ho
        self.s_h = s_ho

        return s_ho_masked, reward, done, info

    def reset(self):
        # Reset env and trace
        super().reset()
        self.trace = list()
        self.s_h = self.stream_learner.graph.initial_state

        return self.s_h

    def sample_rest_of_episode(self):
        # Iterate until episode is over
        policy = self.policy(self.A, self.rng)
        done = False
        while not done:
            # Random action
            action = policy.choose_action()
            # Perform action in the actual environment
            observation, reward, done, _ = super().step(action)
            # Append to histories
            self.trace.append((action, (observation, reward)))
        return True

    def _save_logs(self):
        # Trace lengths
        trace_lengths = dict()
        trace_lengths['trace_lengths'] = self.trace_lengths
        trace_lengths['safe_trace_lengths'] = self.safe_trace_lengths
        update_log(f'{self.log_path}/abstraction', 'trace_lengths', trace_lengths)
        # PDFA Learner stats
        pdfa_learner_stats = self.stream_learner.get_stats()
        update_log(f'{self.log_path}/abstraction', 'pdfa_learner_stats', pdfa_learner_stats)
