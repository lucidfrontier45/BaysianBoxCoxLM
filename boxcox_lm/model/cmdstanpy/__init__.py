import time
from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel

from ..base import BayesianModel

_model_file_path = Path(__file__).parent.joinpath("model.stan")


class CmdStanPyModel(BayesianModel):
    def __init__(
        self,
        w_s: float = 2,
        sigma_beta: float = 2,
        lambda_s: float = 1,
        n_warmup: int = 500,
        n_samples: int = 1000,
        n_chains: int = 4,
    ) -> None:
        self._model_params = {
            "w_s": w_s,
            "sigma_beta": sigma_beta,
            "lambda_s": lambda_s,
        }

        self._mcmc_params = {
            "iter_warmup": n_warmup,
            "iter_sampling": n_samples,
            "chains": n_chains,
        }

        self._model = CmdStanModel(
            "boxcox_lm",
            stan_file=str(_model_file_path),
            stanc_options={"O1": True},
            cpp_options={"STAN_THREADS": True, "O3": True},
        )

    def fit(self, X, y=None, seed=None):
        X = np.array(X)
        if y is not None:
            y = np.array(y)
        if seed is None:
            seed = int(time.time())
        N, D = X.shape
        data = {
            "X": X,
            "y": y,
            "N": N,
            "D": D,
            "X_new": [],
            "N_new": 0,
        } | self._model_params
        self._mcmc = self._model.sample(data=data, **self._mcmc_params)

    def summary(self):
        print(self._mcmc.summary())

    def predict(self, X, seed=None):
        X = np.array(X)
        if seed is None:
            seed = int(time.time())
        N, D = X.shape
        data = {
            "X": [],
            "y": [],
            "N": 0,
            "X_new": X,
            "N_new": N,
            "D": D,
        } | self._model_params
        ret = self._model.generate_quantities(
            data=data, mcmc_sample=self._mcmc, seed=seed
        )
        keys = [f"yp[{i+1}]" for i in range(N)]
        yp = ret.draws_pd()[keys].values
        return yp
