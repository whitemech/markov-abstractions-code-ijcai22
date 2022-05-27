from collections import deque


class RotatingMAB_v2():
    """
    Specification for the RotatingMAB_v2 domain.
    """
    VERSION = 0.1

    def __init__(
        self,
        initial_state: tuple = (0, 0),
        nb_arms: int = 4,
        win_probs: list = [0.9, 0.2, 0.3, 0.1],
        reward_win: int = 100 
    ):
        # States and actions
        self.NB_ARMS = nb_arms
        self.STATES = set([(arm,) for arm in range(nb_arms)])
        self.ACTIONS = set(range(nb_arms))

        # Observations
        self.LOSE = 0
        self.WIN = 1
        self.OBSERVATIONS = set([tuple([self.LOSE]), tuple([self.WIN])])

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
                if observation == tuple([self.WIN]):
                    next_state = (state[0] + 1) % self.NB_ARMS
                    tau_spec[state][observation] = (next_state,)
                else:
                    tau_spec[state][observation] = (0, )
        return tau_spec

    def _generate_theta_spec(self):
        theta_spec = dict()
        for state in self.STATES:
            theta_spec[state] = dict()
            win_probs = deque(self.WIN_PROBABILITIES)
            win_probs.rotate(state[0])
            for action in self.ACTIONS:
                theta_spec[state][action] = dict()
                win_prob = win_probs[action]
                for observation in self.OBSERVATIONS:
                    theta_spec[state][action][observation] = dict()
                    if observation == tuple([self.WIN]):
                        theta_spec[state][action][observation][self.REWARD_WIN] = win_prob
                    else:
                        theta_spec[state][action][observation][self.REWARD_LOSE] = 1 - win_prob
        return theta_spec


def instantiate_env(env_params):
    return RotatingMAB_v2(**env_params)
