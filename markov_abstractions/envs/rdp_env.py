from markov_abstractions.envs.discrete_env import DiscreteEnv
from gym.spaces import Discrete, MultiDiscrete
from collections import defaultdict
from functools import partial
import numpy as np


class RDPEnv(DiscreteEnv):
    """
    RDP environment generated out of specifications of transition function tau and output function theta.
    """
    VERSION = 0.1

    def __init__(
        self, 
        specification, 
        markovian: bool = False, 
        use_stop_prob: bool = False, 
        episode_length: int = None, 
        flicker_prob: float = 0.0,
        rand_seed: int = 0
    ): 
        self.specification = specification
        self.markovian = markovian

        # Initialise random number generator
        self.rng = np.random.default_rng(rand_seed)

        # Set stop probability or fixed episode length (if not None)
        self.use_stop_prob = use_stop_prob
        self.stop_prob = 1/(episode_length+1)
        self.episode_length = episode_length
        self.flicker_prob = flicker_prob

        # Compute space of observations x states
        (
            observation_state_space, 
            observation_space, 
            state_space, 
            observations, 
            states, 
            obs_length, 
            state_length
        ) = self._compute_observation_space(self.tau)
        self._state_length = state_length
        self._obs_length = obs_length
        
        # Encoder/decoder for state symbols
        self.encoder = partial(np.ravel_multi_index, dims=observation_state_space.nvec)
        self.decoder = partial(np.unravel_index, shape=observation_state_space.nvec)

        # Compute space of actions
        action_space, actions = self._compute_action_space(self.theta)
        self.action_space = action_space
        self.actions = actions

        # Compute dynamics
        P, rewards = self._compute_dynamics(self.tau, self.theta, specification.terminal_states)
        self.rewards = rewards

        # Compute initial and terminal states/observations according to markovian/non-markovian
        self.initial_state = self._compute_initial_state(specification.initial_state)
        self.terminal_states = self._compute_terminal_states(specification.terminal_states)

        # Initialise discrete env and set random seed
        nS = len(observations) * len(states)
        nA = len(actions)
        P_initial_state = self.encoder(specification.initial_state)
        ids = np.zeros(nS)
        ids[P_initial_state] = 1.0
        super().__init__(nS, nA, P, ids)
        self.seed(rand_seed)

        # Set correct observation space (replacing the one set from DiscreteEnv in super().__init__ above)
        if markovian: 
            self.observations = states
            if self.flicker_prob > 0:
                self.observations.add(self.flicker_symbol)
            self.observation_space = state_space
        else: 
            self.observations = observations
            if self.flicker_prob > 0:
                self.observations.add(self.flicker_symbol)
            self.observation_space = observation_space

    def _compute_observation_space(self, tau):
        """Compute the space of observations and actual states."""
        # Extract states, observations, and elements of each observation
        states = set()
        observations = set()
        state_elements = defaultdict(set)
        obs_elements = defaultdict(set)
        for state in tau():
            states.add(state)
            for index, state_element in enumerate(state):
                    state_elements[index].add(state_element)
            for observation in tau(state):
                observations.add(observation)
                for index, obs_element in enumerate(observation):
                    obs_elements[index].add(obs_element)
        # Get the number of elements in each state and observation tuples
        # we add +2 to account for terminal and flicker observations
        nb_state_element_values = tuple([len(values)+2 for values in state_elements.values()])
        nb_obs_element_values = tuple([len(values)+2 for values in obs_elements.values()])
        state_length = len(nb_state_element_values)
        obs_length = len(nb_obs_element_values)
        # Generate the spaces
        observation_state_space = MultiDiscrete(nb_obs_element_values + nb_state_element_values)
        observation_space = MultiDiscrete(nb_obs_element_values)
        state_space = MultiDiscrete(nb_state_element_values)
        # Define terminal and flicker observations
        self.terminal_state = tuple(list(state_space.nvec - 2))
        self.terminal_obs = tuple(list(observation_space.nvec - 2))
        self.terminal_symbol = self.terminal_state if self.markovian else self.terminal_obs
        self.flicker_state = tuple(list(state_space.nvec - 1))
        self.flicker_obs = tuple(list(observation_space.nvec - 1))
        self.flicker_symbol = self.flicker_state if self.markovian else self.flicker_obs
        return (
            observation_state_space, 
            observation_space, 
            state_space, 
            observations, 
            states, 
            obs_length, 
            state_length
        )

    def _compute_action_space(self, theta):
        """Compute the action space."""
        # Extract all possible actions from each state
        actions = set()
        for state in theta():
            for action in theta(state):
                actions.add(action)
        # Generate the space
        action_space = Discrete(len(actions))
        return action_space, actions

    def _compute_dynamics(self, tau, theta, terminal_states):
        """Compute the dynamics of system."""
        P = dict()
        rewards = set()
        for state in tau():
            for current_obs in tau(state):
                current_state = tau(state, current_obs)
                s1 = current_obs + current_state
                s1 = self.encoder(s1)
                P[s1] = dict()
                for action in theta(current_state):
                    P[s1][action] = set()
                    for next_obs in theta(current_state, action):
                        next_state = tau(current_state, next_obs)
                        s2 = next_obs + next_state
                        s2 = self.encoder(s2)
                        for reward in theta(current_state, action, next_obs):
                            rewards.add(reward)
                            prob = theta(current_state, action, next_obs, reward)
                            if prob > 0:
                                P[s1][action].add((prob, s2, reward, False))
                    P[s1][action] = list(P[s1][action])
        return P, list(rewards)

    def tau(self, state=None, observation=None):
        if state is None:
            return self.specification.tau_spec
        if observation is None:
            return self.specification.tau_spec[state]
        else:
            return self.specification.tau_spec[state][observation]

    def theta(self, state=None, action=None, observation=None, reward=None):
        if state is None:
            return self.specification.theta_spec
        elif action is None:
            return self.specification.theta_spec[state]
        elif observation is None:
            return self.specification.theta_spec[state][action]
        elif reward is None:
            return self.specification.theta_spec[state][action][observation]
        else:
            return self.specification.theta_spec[state][action][observation][reward]

    def _compute_initial_state(self, initial_state_spec):
        initial_state_spec = list(initial_state_spec)
        if self.markovian: return tuple(initial_state_spec[self._obs_length:])
        else: return tuple(initial_state_spec[:self._obs_length])

    def _compute_terminal_states(self, terminal_states_spec):
        terminal_states = list()
        for state in terminal_states_spec:
            state = list(state)
            if self.markovian: 
                terminal_states.append(tuple(state[self._obs_length:]))
            else: 
                terminal_states.append(tuple(state[:self._obs_length]))
        return set(terminal_states)

    def _process(self, state):
        """Return actual state if Markovian, otherwise observation."""
        observation_and_state = list(self.decoder(state))
        if self.markovian: return tuple(observation_and_state[self._obs_length:])
        else: return tuple(observation_and_state[:self._obs_length])

    def reset(self, **kwargs):
        """Reset the environment."""
        state = super().reset(**kwargs)
        new_state = self._process(state)
        self.steps = 0
        self.in_terminal = False
        return new_state

    def step(self, action):
        """Do a step."""
        # Update number of steps
        self.steps += 1
        # If done next return terminal, else do a step
        if self.in_terminal:
            state = self.terminal_symbol
            reward = 0
            done = True
            info = dict()
        else:
            # Do step
            state, reward, done, info = super().step(action)
            # Process state
            state = self._process(state) 
            # If terminal state, finish at the next step 
            if state in self.terminal_states: 
                self.in_terminal = True 
            # Stop with probability or not
            if self.use_stop_prob:
                if self.rng.random() <= self.stop_prob:
                    self.in_terminal = True
                    done = True
                    if state not in self.terminal_states: 
                        reward = 0
                        state = self.terminal_symbol
            elif self.steps == self.episode_length:
                self.in_terminal = True
                done = True
                if state not in self.terminal_states: 
                    reward = 0
                    state = self.terminal_symbol
            # Flicker or not 
            if self.rng.random() <= self.flicker_prob: 
                state = self.flicker_symbol
        return state, reward, done, info

    def eval_step(self, action):
        """
        Do an evaluation step.
        An evaluation step does not return done with a stop probability,
        it only returns done in actual terminal states.
        """
        # Update number of steps and finish if episode length is set
        self.steps += 1
        # Do step
        state, reward, done, info = super().step(action)
        # Process state 
        state = self._process(state) 
        # If terminal state, finish at the next step 
        if state in self.terminal_states: 
            done = True
        # Flicker or not 
        if self.rng.random() <= self.flicker_prob: 
            state = self.flicker_symbol
        return state, reward, done, info
