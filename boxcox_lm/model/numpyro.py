import time
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist
from jax import grad, random, vmap
from numpy import float32
from numpyro.infer import MCMC, NUTS, Predictive

from .base import BayesianModel


def boxcox(y: float, lam: float):
    return (y**lam - 1.0) / lam


def inv_boxcox(z: float, lam: float):
    return (z * lam + 1.0) ** (1.0 / lam)


def transform(y, lam, m, s):
    return (boxcox(y, lam) - m) / s


trans_grad = vmap(grad(transform), (0, None, 0, None))


def model(
    X: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    w_s=2.0,
    sigma_beta=2.0,
    lambda_s=1.0,
):
    n, d = X.shape
    with pyro.plate("d", d):
        w = pyro.sample("w", dist.Normal(0.0, w_s))
    sigma = pyro.sample("s", dist.Exponential(sigma_beta))
    z = jnp.dot(X, w)
    lam = pyro.sample("lambda", dist.Normal(1.0, lambda_s))
    if y is not None:
        yl = transform(y, lam, z, sigma)
        grad_factor = trans_grad(y, lam, z, sigma)
        pyro.factor("", jnp.log(grad_factor).sum())
        with pyro.plate("n", n):
            pyro.sample("yl", dist.Normal(0, 1), obs=yl)

    else:
        with pyro.plate("n", n):
            yl = pyro.sample("yl", dist.Normal(z, sigma))
            return pyro.deterministic("y", inv_boxcox(yl, lam))


class NumPyroModel(BayesianModel):
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
        self._model = partial(model, w_s=w_s, sigma_beta=sigma_beta, lambda_s=lambda_s)
        self._mcmc = MCMC(
            NUTS(self._model),
            num_warmup=n_warmup,
            num_samples=n_samples,
            num_chains=n_chains,
        )

    def fit(self, X, y=None, seed=None):
        X = jnp.array(X, dtype=float32)
        if y is not None:
            y = jnp.array(y, dtype=float32)
        if seed is None:
            seed = random.PRNGKey(int(time.time()))
        self._mcmc.run(seed, X, y)
        self._predictive = Predictive(self._model, self._mcmc.get_samples())

    def predict(self, X, seed=None):
        X = jnp.array(X, dtype=float32)
        if seed is None:
            seed = random.PRNGKey(int(time.time()))
        return self._predictive(seed, X, y=None)["y"]

    def summary(self):
        self._mcmc.print_summary()
