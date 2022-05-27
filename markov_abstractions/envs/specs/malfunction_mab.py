from itertools import product


class MalfunctionMAB():
    """
    Specification for the Malfunction MAB domain.

    Reference:
    Abadi and Brafman (2020). "Learning and Solving Regular Decision Processes."
    """
    VERSION = 0.1

    def __init__(
        self,
        initial_state: tuple = (0, 0),
        nb_arms: int = 2,
        win_probs: list = [0.8, 0.2],
        malfunction_arm: int = 0,
        malfunction_count: int = 5,
        reward_win: int = 100 
    ):
        # Malfunction settings
        self.malfunction_arm = malfunction_arm
        self.malfunction_count = malfunction_count

        # States and actions
        self.NB_ARMS = nb_arms
        self.STATES = set([(count,) for count in range(malfunction_count + 1)])
        self.ACTIONS = set(range(nb_arms))

        # Observations
        self.LOSE = 0 
        self.WIN = 1 
        self.OBSERVATIONS = set([(action,) for action in self.ACTIONS]) 

        # Observation probabilities
        self.WIN_PROBABILITIES = win_probs

        # Rewards
        self.REWARD_LOSE = 0
        self.REWARD_WIN = reward_win
        self.REWARDS = set([self.REWARD_LOSE, self.REWARD_WIN])

        # Specify the initial state
        self.initial_state = initial_state

        # Specify terminal states (if any)
        self.terminal_states = set()

        # Specify transition function and output function
        self.tau_spec = self._generate_tau_spec()
        self.theta_spec = self._generate_theta_spec()

    def _generate_tau_spec(self):
        tau_spec = dict()
        for state in self.STATES:
            tau_spec[state] = dict()
            for observation in self.OBSERVATIONS:
                action = observation[0]
                # Increase counter
                if action == self.malfunction_arm or state[0] == self.malfunction_count:
                    next_state = (state[0] + 1) % (self.malfunction_count + 1)
                else:
                    next_state = state[0]
                tau_spec[state][observation] = (next_state,)
        return tau_spec

    def _generate_theta_spec(self):
        theta_spec = dict()
        for state in self.STATES:
            theta_spec[state] = dict()
            for action in self.ACTIONS:
                theta_spec[state][action] = dict()
                for observation in self.OBSERVATIONS:
                    if observation[0] == action:
                        # If state is broken arm and presses the broken arm, then prob of winning is 0
                        if state[0] == self.malfunction_count and action == self.malfunction_arm:
                            win_prob = 0.0
                        else:
                            win_prob = self.WIN_PROBABILITIES[action]
                        # Set win probability
                        theta_spec[state][action][observation] = dict()
                        theta_spec[state][action][observation][self.REWARD_WIN] = win_prob
                        theta_spec[state][action][observation][self.REWARD_LOSE] = 1 - win_prob
        return theta_spec


def instantiate_env(env_params):
    return MalfunctionMAB(**env_params)
