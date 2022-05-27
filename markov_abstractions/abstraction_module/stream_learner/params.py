import pprint 


class StreamParams:
    """
    Parameters for the StreamPDFALearner (Balle et al., 2013) algorithm.

    Gamma: the alphabet set Gamma (actions).
    Sigma: the alphabet set Sigma (observations and rewards).
    epsilon: accuracy.
    delta: confidence.
    mu: distinguishability.
    n: number of states.

    Other parameters defined in terms of the ones above:

    theta: insignificance threshold
    delta_p: conficence parameter
    alpha: milestones for significance testing
    alpha_0: milestones for significance testing
    """
    def __init__(self, **kwargs):
        self.d = kwargs["d"]
        self.nb_actions = kwargs["nb_actions"]
        self.nb_observations = kwargs["nb_observations"]
        self.nb_rewards = kwargs["nb_rewards"]

        self.Gamma: set = kwargs["Gamma"]
        self.Sigma: set = kwargs["Sigma"]
        self.alphabet_size = len(self.Gamma) * len(self.Sigma)

        self.epsilon: float = kwargs["epsilon"]
        self.delta: float = kwargs["delta"]
        self.mu: float = kwargs["mu"]
        self.n: int = kwargs["n"]

        self.delta_p: float = self.delta / (2 * self.alphabet_size * self.n * (self.n + 2))
        self.alpha: float = kwargs["alpha"]  # 2
        self.alpha_0: int = kwargs["alpha_0"]  # 128

        self.string_propagation: bool = kwargs["string_propagation"]
        self.log_path: str = kwargs["log_path"]
       
    def __repr__(self):
        """Get the representation."""
        return pprint.pformat(
            {
                "Gamma": self.Gamma,
                "Sigma": self.Sigma,
                "alphabet_size": self.alphabet_size,
                "d": self.d,
                "epsilon": self.epsilon,
                "delta": self.delta,
                "mu": self.mu,
                "n": self.n,
                "alpha": self.alpha,
                "alpha_0": self.alpha_0,
                "delta_p": self.delta_p,
                "string_propagation": self.string_propagation
            }
        )
