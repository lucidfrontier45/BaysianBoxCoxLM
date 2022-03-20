from abc import ABC, abstractmethod
from typing import Optional


class BayesianModel(ABC):
    @abstractmethod
    def __init__(
        self,
        w_s: float = 2.0,
        sigma_beta: float = 2.0,
        lambda_s: float = 1.0,
        n_warmup: int = 500,
        n_samples: int = 1000,
        n_chains: int = 4,
    ) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, X, y=None, seed=None):
        pass

    @abstractmethod
    def predict(self, X, seed=None):
        pass

    @abstractmethod
    def summary(self):
        pass
