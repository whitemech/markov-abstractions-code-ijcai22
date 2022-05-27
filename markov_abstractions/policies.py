import numpy as np


class RandomPolicy:

    def __init__(self, actions, rng):
        # Initialise random number generator
        self.rng = rng
        # Actions
        self.actions = actions

    def choose_action(self):
        action = self.rng.integers(0, len(self.actions))
        return action


class SingleRandomActionPolicy:

    def __init__(self, actions, rng):
        # Initialise random number generator
        self.rng = rng
        # Actions
        self.actions = actions
        # Action to take
        self.action = self.rng.integers(0, len(self.actions))

    def choose_action(self):
        return self.action


class SingleActionPolicy:

    def __init__(self, action):
        # Action to take
        self.action = action

    def choose_action(self):
        return self.action
