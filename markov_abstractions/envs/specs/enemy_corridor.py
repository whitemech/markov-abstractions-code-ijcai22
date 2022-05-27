from itertools import product


class EnemyCorridor():
    """
    Specification for the Enemy Corridor domain.

    Reference:
    Ronca and De Giacomo (2021). "Efficient PAC Reinforcement Learning in Regular Decision Processes."
    """
    VERSION = 0.1

    def __init__(
        self,
        m: int = 4,
        probs: list = [0.2, 0.9],
    ):
        # Actions
        self.UP = 0
        self.DOWN = 1
        self.ACTIONS = set([self.UP, self.DOWN])

        # States
        self.M = m
        self.STATES = set()
        for m, current_prob in product(range(self.M), range(2)):
            self.STATES.add((m, current_prob))
        
        # Observations 
        self.OBSERVATIONS = set([(m, enemy) for m, enemy in product(range(self.M), range(2))]) 

        # Probabilities
        self.probs = probs 

        # Rewards
        self.REWARD_ENEMY = 0
        self.REWARD_CLEAR = 100
        self.REWARDS = set([self.REWARD_ENEMY, self.REWARD_CLEAR])

        # Specify the initial state (tuple concatenating initial observation and state)
        self.initial_state = tuple([0, 0, 0, 0])

        # Specify terminal states (if any)
        self.terminal_states = set()

        # Specify transition function and output function
        self.tau_spec = self._generate_tau_spec()
        self.theta_spec = self._generate_theta_spec() 

    def _generate_tau_spec(self):
        tau = dict()
        for state in self.STATES:
            tau[state] = dict()
            state_m = state[0]
            state_prob = state[1]
            for observation in self.OBSERVATIONS:
                obs_m = observation[0]
                obs_enemy = observation[1]
                # If the observed m is a correct next value for previous state's m
                if (state_m+1) % self.M == obs_m:
                    # If observed enemy, invert prob
                    if obs_enemy == 1:
                        next_prob = (state_prob+1) % 2
                    else:  # Keep using the same probability
                        next_prob = state_prob
                    next_state = (obs_m, next_prob)
                    tau[state][observation] = next_state
        return tau

    def _generate_theta_spec(self):
        theta = dict()
        for state in self.STATES:
            theta[state] = dict()
            state_m = state[0]
            state_prob = state[1]
            for action in self.ACTIONS:
                theta[state][action] = dict()
                for observation in self.OBSERVATIONS:
                    obs_m = observation[0]
                    obs_enemy = observation[1]
                    # If  observed m is a correct next value for previous state's m
                    if (state_m+1) % self.M == obs_m:
                        theta[state][action][observation] = dict()
                        # If it's in half of the corridor, change probability set
                        if (state_m+1) % self.M >= self.M/2:
                            next_prob_set = 1
                        else:  # Keep using the same probability set
                            next_prob_set = 0
                        # Get probability of enemy
                        if state_prob == 0:
                            if action == 0:
                                enemy_prob = self.probs[next_prob_set]
                            else:
                                enemy_prob = 1 - self.probs[next_prob_set]
                        else:
                            if action == 0:
                                enemy_prob = 1 - self.probs[next_prob_set]
                            else:
                                enemy_prob = self.probs[next_prob_set]
                        # If observed enemy
                        if obs_enemy == 1:
                            theta[state][action][observation][self.REWARD_ENEMY] = enemy_prob
                        else:
                            theta[state][action][observation][self.REWARD_CLEAR] = 1 - enemy_prob
        return theta


def instantiate_env(env_params):
    return EnemyCorridor(**env_params)
