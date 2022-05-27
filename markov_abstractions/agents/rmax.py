from markov_abstractions.utils import update_log

from collections import defaultdict, Counter
from math import ceil, log
from copy import deepcopy
import numpy as np


class RMaxAgent():

    def __init__(
        self, 
        env, 
        stop_prob, 
        epsilon=0.05, 
        delta=0.05, 
        m_0=None, 
        rand_seed=0, 
        use_vmax=True, 
        evaluation_frequency=10000, 
        evaluation_episodes=50, 
        evaluation_steps=10,
        log_frequency=10000,
        log_path='log'
    ):
        # Initialise random number generator
        self.rng = np.random.default_rng(rand_seed)

        # Environment
        self.env = env
        self.initial_state = self.env.reset()
        self.terminal_state = self.env.terminal_symbol
        self.O = list(env.observations)
        self.A = list(env.actions)
        self.R_max = np.max(env.rewards)

        # Params
        self.epsilon = epsilon  # accuracy 
        self.delta = delta  # confidence 

        # Set Vmax
        if use_vmax:
            self.V_max = self.R_max/stop_prob 
        else: 
            self.V_max = self.R_max 

        # Minimum count before being confident about a tuple 
        self.C = 1
        if m_0 == None:
            self.m_0 = self.C * self.V_max**2 * ((len(self.O) + log((len(self.A) * len(self.O)) / delta)) / (epsilon**2 * stop_prob**2))  
        else:
            self.m_0 = m_0

        # Number of iterations to compute an epsilon-optimal policy 
        self.m_1 = ceil((log(8 / (epsilon * stop_prob))) / stop_prob) 

        # Initialise estimates
        self.init()

        # Evaluation
        self.evaluation_frequency = evaluation_frequency
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_steps = evaluation_steps

        # Stats
        self.evaluations = dict()
        self.training = dict()
        self.episode_rewards = list()
        self.episode_steps = list()
        self.termination_errors = list()
        self.log_path = log_path
        self.log_frequency = log_frequency
        
    def init(self):
        # Initialise estimates
        self.q = defaultdict(lambda: np.zeros(len(self.A)))  # action-value function
        self.S = set()  # set of visited states
        self.reward = dict()  # reward accumulator for each tuple (s,a,s')
        self.count = Counter()  # counts tuples of (s,a) and (s,a,s')
        # Initialise value function for initial state
        self.S = self.S.union([self.initial_state])
        for a in self.A:
            self.q[self.terminal_state][a] = 0
            self.q[self.initial_state][a] = self.V_max
            self.reward[(self.initial_state, a, self.initial_state)] = 0
            self.reward[(self.initial_state, a, self.terminal_state)] = 0 

    def train(self, nb_episodes=1000000):
        for episode in range(nb_episodes):
            # Stats
            self.rewards = 0
            self.steps = 0
            # Reset environment
            state = self.env.reset()
            done = False
            while not done:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                # Observe
                self.observe(state, action, reward, next_state)
                state = next_state
            # Stats
            self.episode_rewards.append(self.rewards)
            self.episode_steps.append(self.steps)
            if episode % self.evaluation_frequency == 0:
                self._perform_evaluation(episode)
            if episode % self.log_frequency == 0:
                self._save_logs()

    def train_on_abstraction(self, nb_episodes=1000000):
        # Start training
        for episode in range(nb_episodes):
            # Stats
            self.rewards = 0
            self.steps = 0
            # Reset environment
            state = self.env.reset()
            done = False
            while not done:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action, self)
                # Break if done, otherwise observe
                if done: 
                    break
                else:
                    self.observe(state, action, reward, next_state)
                    state = next_state
            # Stats
            self.episode_rewards.append(self.rewards)
            self.episode_steps.append(self.steps)
            if episode % self.evaluation_frequency == 0:
                self._perform_evaluation_abstraction(episode)
            if episode % self.log_frequency == 0:
                self._save_logs()
        
    def choose_action(self, state):
        tie = np.all(self.q[state] == self.q[state][0])
        if tie: return self.rng.integers(0, len(self.A))
        else: return np.argmax(self.q[state])

    def observe(self, state, action, reward, next_state):
        # Stats
        self.steps += 1
        self.rewards += reward
        # If new state, add it and initialise it optimistically
        if next_state not in self.S and next_state != self.terminal_state:
            # Add new state to the set
            self.S = self.S.union([next_state])
            for a in self.A:
                # Initialise value function estimates
                self.q[next_state][a] = self.V_max
                # Initialise reward estimates
                self.reward[(next_state, a, self.terminal_state)] = 0
                for s in self.S:
                    self.reward[(next_state, a, s)] = 0
                    self.reward[(s, a, next_state)] = 0
        
        # If transition is below minimum count, increment estimates
        if self.count[(state,action)] < self.m_0:
            self.count.update([(state,action)])
            self.count.update([(state,action,next_state)])
            self.reward[(state,action,next_state)] += reward
            # If transition reaches minimum count, update value function
            if self.count[(state, action)] == self.m_0:
                self._update_value_function()
                
    def _update_value_function(self):
        for _ in range(self.m_1):
            q = deepcopy(self.q)
            for s in self.S:
                for a in self.A:
                    if self.count[(s,a)] >= self.m_0:
                        q[s][a] = 0
                        states = self.S.union((self.terminal_state,))
                        for s_p in states:
                            trans_prob = self.count[(s,a,s_p)] / self.count[(s,a)]
                            if trans_prob > 0:
                                reward = self.reward[(s,a,s_p)] / self.count[(s,a,s_p)]
                                q[s][a] += trans_prob * (reward + np.max(self.q[s_p]))
            self.q = q
        
    def merge_states(self, merges):
        if merges:
            # Iterate merges and adjust counts
            updated_counter = False
            to_remove = set()
            for (s_merged, s_to) in merges:
                if s_merged in self.S and s_to in self.S:
                    for s in self.S:
                        for a in self.A:
                            if self.count[(s,a,s_merged)] > 0:
                                self.count[(s,a,s_to)] += self.count[(s,a,s_merged)]
                                self.reward[(s,a,s_to)] += self.reward[(s,a,s_merged)]
                                updated_counter = True
                            to_remove.add((s,a,s_merged))
            # Remove merged states' estimates
            merged = set()
            for (s, a, s_merged) in to_remove:
                merged.add(s_merged)
                del self.count[(s,a,s_merged)]
                del self.reward[(s,a,s_merged)]
            for s in merged:
                self.S = self.S.difference([s])
                del self.q[s]
            # Update value function
            if updated_counter:
                self._update_value_function()
    
    def termination_test(self):
        # Get value function for evaluation
        q_0 = self._compute_evaluation_value_function()
        
        # Compute the error
        max_q_0 = np.max(q_0[self.initial_state]) 
        max_q = np.max(self.q[self.initial_state]) 
        error = abs(max_q_0 - max_q) 
        
        # Stats
        self.termination_errors.append(error)

        # Evaluate error
        if error < self.epsilon/4: return True
        else: return False

    def _compute_evaluation_value_function(self):
        # Copy value function but set zero for pairs below threshold m_0
        q_0 = deepcopy(self.q)
        for s in self.S:
            for a in self.A:
                if self.count[(s,a)] < self.m_0:
                    q_0[s][a] = 0
        # Update value function copy
        for _ in range(self.m_1):
            q = deepcopy(q_0)
            for s in self.S:
                for a in self.A:
                    if self.count[(s,a)] >= self.m_0:
                        q[s][a] = 0
                        states = self.S.union((self.terminal_state,))
                        for s_p in states:
                            trans_prob = self.count[(s,a,s_p)] / self.count[(s,a)]
                            if trans_prob > 0:
                                reward = self.reward[(s,a,s_p)] / self.count[(s,a,s_p)]
                                q[s][a] += trans_prob * (reward + np.max(q_0[s_p]))
            q_0 = q
        return q_0

    def _perform_evaluation(self, evaluation_episode):
        # Get value function for evaluation
        q = self._compute_evaluation_value_function()
        # Stats
        evaluation_stats = dict()
        evaluation_stats['episode_rewards'] = list()
        evaluation_stats['episode_steps'] = list()
        # Start evaluating
        for episode in range(self.evaluation_episodes):
            # Stats
            # episode_start = time.time()
            rewards = 0
            steps = 0
            # Reset environment
            state = self.env.reset()
            for step in range(self.evaluation_steps):
                # Choose and take action
                action = self._choose_evaluation_action(state, q)
                next_state, reward, done, _ = self.env.eval_step(action)
                # Stats
                rewards += reward
                steps += 1
                # Break if done
                if done:
                    break
                else:
                    state = next_state
            # Stats
            evaluation_stats['episode_rewards'].append(rewards)
            evaluation_stats['episode_steps'].append(steps)
        # Stats
        self.evaluations[evaluation_episode] = evaluation_stats

    def _perform_evaluation_abstraction(self, evaluation_episode):
        # Get value function for evaluation
        q = self._compute_evaluation_value_function()
        # Stats
        evaluation_stats = dict()
        evaluation_stats['episode_rewards'] = list()
        evaluation_stats['episode_steps'] = list()
        # Start evaluating
        for episode in range(self.evaluation_episodes):
            # Stats
            rewards = 0
            steps = 0
            # Reset environment
            state = self.env.reset()
            for step in range(self.evaluation_steps):
                # Choose and take action
                action = self._choose_evaluation_action(state, q)
                next_state, reward, done, _ = self.env.evaluation_step(action)
                # Stats
                rewards += reward
                steps += 1
                # Break if done
                if done:
                    break
                else:
                    state = next_state
            # Stats
            evaluation_stats['episode_rewards'].append(rewards)
            evaluation_stats['episode_steps'].append(steps)
        # Stats
        self.evaluations[evaluation_episode] = evaluation_stats

    def _choose_evaluation_action(self, state, q):
        if state is not None and state in q.keys():
            tie = np.all(q[state] == q[state][0])
            if tie: return self.rng.integers(0, len(self.A))
            else: return np.argmax(self.q[state])
        else:
            return self.rng.integers(0, len(self.A))

    def _save_logs(self):
        # Training and evaluation
        update_log(self.log_path, 'training_episode_reward', self.episode_rewards)
        update_log(self.log_path, 'training_episode_steps', self.episode_steps)
        update_log(self.log_path, 'evaluations', dict(self.evaluations))
        # RMax data
        update_log(f'{self.log_path}/rmax', 'q', dict(self.q))
        update_log(f'{self.log_path}/rmax', 'count', dict(self.count))
        update_log(f'{self.log_path}/rmax', 'reward', dict(self.reward))
        update_log(f'{self.log_path}/rmax', 'termination_errors', self.termination_errors)
