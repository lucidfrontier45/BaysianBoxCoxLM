import enum

from .base import BayesianModel
from .cmdstanpy import CmdStanPyModel
from .numpyro import NumPyroModel


class BackendType(enum.Enum):
    NumPyro = "numpyro"
    CmdStanPy = "cmdstanpy"


_backends = {BackendType.NumPyro: NumPyroModel, BackendType.CmdStanPy: CmdStanPyModel}


class BayesianBoxCoxLinearRegression(BayesianModel):
    def __init__(
        self,
        w_s: float = 2,
        sigma_beta: float = 2,
        lambda_s: float = 1,
        n_warmup: int = 500,
        n_samples: int = 1000,
        n_chains: int = 4,
        backend: BackendType = BackendType.NumPyro,
    ) -> None:
        self._backend = _backends[backend](
            w_s, sigma_beta, lambda_s, n_warmup, n_samples, n_chains
        )

    def fit(self, X, y=None, seed=None):
        return self._backend.fit(X, y, seed)

    def predict(self, X, seed=None):
        return self._backend.predict(X, seed)

    def summary(self):
        return self._backend.summary()
