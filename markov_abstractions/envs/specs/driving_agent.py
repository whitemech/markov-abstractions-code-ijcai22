

class DrivingAgent():
    """
    Specification for the Driving Agent domain.
    """
    VERSION = 0.1
    
    def __init__(
        self, 
        accident_prob: float = 1.0,
        initial_state: tuple = (0, 0, 0)
    ):
        # States
        DRY = 0
        WET = 1

        # Observations
        SUNNY = 0
        CLOUDY = 1
        RAINY = 2
        #
        NO_ACCIDENT = 0
        ACCIDENT = 1

        # Observation probabilities
        SUNNY_PROBABILITY = 0.2
        CLOUDY_PROBABILITY = 0.6
        RAINY_PROBABILITY = 0.2
        #
        ACCIDENT_PROBABILITY = accident_prob
        NO_ACCIDENT_PROBABILITY = 1 - accident_prob

        # Actions
        DRIVE_NORMAL = 0
        DRIVE_SLOW = 1

        # Rewards
        REWARD_NORMAL = 20
        REWARD_SLOW = 18
        REWARD_ACCIDENT = 0

        # Specify the initial state
        self.initial_state = initial_state

        # Specify terminal states (if any)
        self.terminal_states = set()

        # Specify transition function
        self.tau_spec = {
            (DRY,): {
                (SUNNY, NO_ACCIDENT): (DRY,),
                (CLOUDY, NO_ACCIDENT): (DRY,),
                (RAINY, NO_ACCIDENT): (WET,),
            },
            (WET,): {
                (SUNNY, NO_ACCIDENT): (DRY,),
                (CLOUDY, NO_ACCIDENT): (WET,),
                (RAINY, NO_ACCIDENT): (WET,),
                (SUNNY, ACCIDENT): (DRY,),
                (CLOUDY, ACCIDENT): (WET,),
                (RAINY, ACCIDENT): (WET,),
            }
        }

        # Specify output function
        self.theta_spec = {
            (DRY,): {
                DRIVE_NORMAL: {
                    (SUNNY, NO_ACCIDENT): {
                        REWARD_NORMAL: SUNNY_PROBABILITY,
                        REWARD_SLOW: 0.0,
                        REWARD_ACCIDENT: 0.0
                    },
                    (CLOUDY, NO_ACCIDENT): {
                        REWARD_NORMAL: CLOUDY_PROBABILITY,
                        REWARD_SLOW: 0.0,
                        REWARD_ACCIDENT: 0.0
                    },
                    (RAINY, NO_ACCIDENT): {
                        REWARD_NORMAL: RAINY_PROBABILITY,
                        REWARD_SLOW: 0.0,
                        REWARD_ACCIDENT: 0.0
                    }
                },
                DRIVE_SLOW: {
                    (SUNNY, NO_ACCIDENT): {
                        REWARD_NORMAL: 0.0,
                        REWARD_SLOW: SUNNY_PROBABILITY,
                        REWARD_ACCIDENT: 0.0
                    },
                    (CLOUDY, NO_ACCIDENT): {
                        REWARD_NORMAL: 0.0,
                        REWARD_SLOW: CLOUDY_PROBABILITY,
                        REWARD_ACCIDENT: 0.0
                    },
                    (RAINY, NO_ACCIDENT): {
                        REWARD_NORMAL: 0.0,
                        REWARD_SLOW: RAINY_PROBABILITY,
                        REWARD_ACCIDENT: 0.0
                    }
                }
            },
            (WET,): {
                DRIVE_NORMAL: {
                    (SUNNY, NO_ACCIDENT): {
                        REWARD_NORMAL: SUNNY_PROBABILITY * NO_ACCIDENT_PROBABILITY,
                        REWARD_SLOW: 0.0,
                        REWARD_ACCIDENT: 0.0
                    },
                    (CLOUDY, NO_ACCIDENT): {
                        REWARD_NORMAL: CLOUDY_PROBABILITY * NO_ACCIDENT_PROBABILITY,
                        REWARD_SLOW: 0.0,
                        REWARD_ACCIDENT: 0.0
                    },
                    (RAINY, NO_ACCIDENT): {
                        REWARD_NORMAL: RAINY_PROBABILITY * NO_ACCIDENT_PROBABILITY,
                        REWARD_SLOW: 0.0,
                        REWARD_ACCIDENT: 0.0
                    },
                    (SUNNY, ACCIDENT): {
                        REWARD_NORMAL: 0.0,
                        REWARD_SLOW: 0.0,
                        REWARD_ACCIDENT: SUNNY_PROBABILITY * ACCIDENT_PROBABILITY
                    },
                    (CLOUDY, ACCIDENT): {
                        REWARD_NORMAL: 0.0,
                        REWARD_SLOW: 0.0,
                        REWARD_ACCIDENT: CLOUDY_PROBABILITY * ACCIDENT_PROBABILITY
                    },
                    (RAINY, ACCIDENT): {
                        REWARD_NORMAL: 0.0,
                        REWARD_SLOW: 0.0,
                        REWARD_ACCIDENT: RAINY_PROBABILITY * ACCIDENT_PROBABILITY
                    }
                },
                DRIVE_SLOW: {
                    (SUNNY, NO_ACCIDENT): {
                        REWARD_NORMAL: 0.0,
                        REWARD_SLOW: SUNNY_PROBABILITY,
                        REWARD_ACCIDENT: 0.0
                    },
                    (CLOUDY, NO_ACCIDENT): {
                        REWARD_NORMAL: 0.0,
                        REWARD_SLOW: CLOUDY_PROBABILITY,
                        REWARD_ACCIDENT: 0.0
                    },
                    (RAINY, NO_ACCIDENT): {
                        REWARD_NORMAL: 0.0,
                        REWARD_SLOW: RAINY_PROBABILITY,
                        REWARD_ACCIDENT: 0.0
                    }
                }
            }
        }


def instantiate_env(env_params):
    return DrivingAgent(**env_params)
