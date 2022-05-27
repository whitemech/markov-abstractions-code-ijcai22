from itertools import product


class Grid():
    """
    Specification for a Grid domain.
    """
    VERSION = 0.1

    def __init__(
        self,
        initial_state: tuple = (0, 0, 0, 0),
        goal_position: list = [3,4],
        n: int = 8,
        m: int = 8,
        prob_success_move: float = 1.0,
        reward_goal: int = 100 
    ):
        # Actions
        self.UP = 0
        self.LEFT = 1
        self.DOWN = 2
        self.RIGHT = 3
        self.ACTIONS = set([self.UP, self.LEFT, self.DOWN, self.RIGHT])

        # Action probabilities
        self.PROB_SUCCESS_MOVE = prob_success_move
        self.PROB_OPPOSITE_MOVE = 1 - prob_success_move 

        # States
        self.GRID_N = n
        self.GRID_M = m
        self.POSITIONS = set([(n, m) for n, m in product(range(self.GRID_N), range(self.GRID_M))])
        self.STATES = self.POSITIONS
        self.GOAL_POSITION = tuple(goal_position)

        # Observations
        self.OBSERVATIONS = set([(position[0], position[1]) for position in self.POSITIONS])

        # Rewards
        self.REWARD_STEP = 0
        self.REWARD_GOAL = reward_goal
        self.REWARDS = set([self.REWARD_STEP, self.REWARD_GOAL])

        # Specify the initial state
        self.initial_state = initial_state

        # Specify terminal states (if any)
        self.terminal_states = self._compute_terminal_states()

        # Specify transition function and output function
        self.tau_spec = self._generate_tau_spec()
        self.theta_spec = self._generate_theta_spec()

    def _compute_terminal_states(self):
        terminal_states = set()
        for state in self.STATES:
            for obs in self.OBSERVATIONS:
                if (
                    state[0] == self.GOAL_POSITION[0] and state[1] == self.GOAL_POSITION[1] and
                    obs[0] == self.GOAL_POSITION[0] and obs[1] == self.GOAL_POSITION[1]
                ):
                    terminal_states.add(tuple(list(obs) + list(state)))
        return terminal_states

    def _next_position_is_valid(self, prev_pos, next_pos):
        prev_x = prev_pos[0]
        prev_y = prev_pos[1]
        move_up = (
            (next_pos == (prev_x, prev_y + 1)) or 
            (next_pos == prev_pos and prev_y == self.GRID_N-1)
        )
        move_left = (
            (next_pos == (prev_x - 1, prev_y)) or
            (next_pos == prev_pos and prev_x == 0)
        )
        move_down = (
            (next_pos == (prev_x, prev_y - 1)) or
            (next_pos == prev_pos and prev_y == 0)
        )
        move_right = (
            (next_pos == (prev_x + 1, prev_y)) or
            (next_pos == prev_pos and prev_x == self.GRID_M-1)
        )
        return move_up or move_left or move_down or move_right

    def _transition_is_valid(self, prev_pos, next_pos, action):
        prev_x = prev_pos[0]
        prev_y = prev_pos[1]

        move_up = (
            (action == self.UP and next_pos == (prev_x, prev_y + 1)) or 
            (action == self.UP and next_pos == prev_pos and prev_y == self.GRID_N-1)
        )
        move_left = (
            (action == self.LEFT and next_pos == (prev_x - 1, prev_y)) or
            (action == self.LEFT and next_pos == prev_pos and prev_x == 0)
        )
        move_down = (
            (action == self.DOWN and next_pos == (prev_x, prev_y - 1)) or
            (action == self.DOWN and next_pos == prev_pos and prev_y == 0)
        )
        move_right = (
            (action == self.RIGHT and next_pos == (prev_x + 1, prev_y)) or
            (action == self.RIGHT and next_pos == prev_pos and prev_x == self.GRID_M-1)
        )
        condition = move_up or move_left or move_down or move_right
        return condition

    def _generate_tau_spec(self):
        tau_spec = dict()
        for state in self.STATES:
            tau_spec[state] = dict()
            prev_pos_x = state[0]
            prev_pos_y = state[1]
            for observation in self.OBSERVATIONS:
                next_pos_x = observation[0]
                next_pos_y = observation[1]
                if self._next_position_is_valid((prev_pos_x, prev_pos_y), (next_pos_x, next_pos_y)):
                    next_state = (next_pos_x, next_pos_y) 
                    tau_spec[state][observation] = next_state
        return tau_spec

    def _generate_theta_spec(self):
        theta_spec = dict()
        for state in self.STATES:
            theta_spec[state] = dict()
            prev_pos_x = state[0]
            prev_pos_y = state[1]
            for action in self.ACTIONS:
                theta_spec[state][action] = dict()
                if action == self.UP: opposite_action = self.DOWN
                elif action == self.LEFT: opposite_action = self.RIGHT
                elif action == self.DOWN: opposite_action = self.UP
                elif action == self.RIGHT: opposite_action = self.LEFT
                for observation in self.OBSERVATIONS:
                    obs_pos_x = observation[0]
                    obs_pos_y = observation[1]
                    # Observation for successful move
                    if self._transition_is_valid((prev_pos_x, prev_pos_y), (obs_pos_x, obs_pos_y), action):
                        theta_spec[state][action][observation] = dict()
                        if (obs_pos_x, obs_pos_y) == self.GOAL_POSITION:
                            theta_spec[state][action][observation][self.REWARD_GOAL] = self.PROB_SUCCESS_MOVE
                        else:
                            theta_spec[state][action][observation][self.REWARD_STEP] = self.PROB_SUCCESS_MOVE
                    # Observation for moving the opposite direction
                    if self._transition_is_valid((prev_pos_x, prev_pos_y), (obs_pos_x, obs_pos_y), opposite_action):
                        theta_spec[state][action][observation] = dict()
                        if (obs_pos_x, obs_pos_y) == self.GOAL_POSITION:
                            theta_spec[state][action][observation][self.REWARD_GOAL] = self.PROB_OPPOSITE_MOVE
                        else:
                            theta_spec[state][action][observation][self.REWARD_STEP] = self.PROB_OPPOSITE_MOVE
        return theta_spec


def instantiate_env(env_params):
    return Grid(**env_params)
