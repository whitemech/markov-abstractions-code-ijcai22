from itertools import product


class CheatMAB():
    """
    Specification for the Cheat MAB domain.

    Reference:
    Abadi and Brafman (2020). "Learning and Solving Regular Decision Processes."
    """
    VERSION = 0.1

    def __init__(
        self,
        initial_state: tuple = (0, 0),
        nb_arms: int = 2,
        win_probs: list = [0.2, 0.2],
        cheat_sequence: list = [0, 0, 0, 1],
        reward_win: int = 100 
    ):
        # Cheat sequence
        self.cheat_sequence = cheat_sequence

        # States and actions
        self.NB_ARMS = nb_arms
        self.STATES = set([(seq_index,) for seq_index in range(len(cheat_sequence) + 1)])
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
                if state[0] == len(self.cheat_sequence):
                    next_state = state[0]
                    tau_spec[state][observation] = (next_state,)
                else:
                    for i, cheat_arm in enumerate(self.cheat_sequence):
                        if state[0] == i:
                            if action == cheat_arm:
                                next_state = state[0] + 1
                                break
                            else:
                                next_state = 0
                                break
                    tau_spec[state][observation] = (next_state,)
        return tau_spec

    def _generate_theta_spec(self):
        theta_spec = dict()
        for state in self.STATES:
            theta_spec[state] = dict()
            for action in self.ACTIONS:
                theta_spec[state][action] = dict()
                for observation in self.OBSERVATIONS:
                    if action == observation[0]:
                        if state[0] == len(self.cheat_sequence):
                            theta_spec[state][action][observation] = dict()
                            theta_spec[state][action][observation][self.REWARD_WIN] = 1.0
                            theta_spec[state][action][observation][self.REWARD_LOSE] = 0.0
                        else:
                            theta_spec[state][action][observation] = dict()
                            theta_spec[state][action][observation][self.REWARD_WIN] = self.WIN_PROBABILITIES[action]
                            theta_spec[state][action][observation][self.REWARD_LOSE] = 1 - self.WIN_PROBABILITIES[action]
        return theta_spec


def instantiate_env(env_params):
    return CheatMAB(**env_params)
