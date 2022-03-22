import time
from typing import Optional

import numpy as np
import numpyro
import typer

from boxcox_lm import BackendType, BayesianBoxCoxLinearRegression

app = typer.Typer(name="baysian_boxcox_regression", add_completion=False)


@app.command()
def main(
    n_data: int = typer.Option(50, help="number of input data"),
    w_s: float = typer.Option(2.0, help="parameter of the w prior"),
    sigma_beta: float = typer.Option(2.0, help="parameter of the sigma prior"),
    lambda_s: float = typer.Option(2.0, help="parameter of the lambda prior"),
    n_warmup: int = typer.Option(500, help="MCMC warmup samples"),
    n_samples: int = typer.Option(1000, help="MCMC samples"),
    n_chains: int = typer.Option(4, help="MCMC chains"),
    seed: Optional[int] = None,
    backend: BackendType = typer.Option(BackendType.NumPyro, case_sensitive=False),
):

    if seed is None:
        seed = int(time.time())

    if backend == BackendType.NumPyro:
        import multiprocessing

        import jax.random

        numpyro.set_host_device_count(multiprocessing.cpu_count())
        key = jax.random.PRNGKey(seed)
        train_seed, predict_seed = jax.random.split(key)
    else:
        train_seed = seed
        predict_seed = seed + 100

    np.random.seed(seed=seed)
    N = n_data
    D = 2

    w = np.array([5.0, -1.2])
    X = np.random.randn(N, D - 1) * 0.1
    X = np.column_stack([np.ones(N), X])
    z = np.dot(X, w)
    y = np.array([np.random.poisson(zz) for zz in np.exp(z)])

    model = BayesianBoxCoxLinearRegression(
        w_s=w_s,
        sigma_beta=sigma_beta,
        lambda_s=lambda_s,
        n_warmup=n_warmup,
        n_samples=n_samples,
        n_chains=n_chains,
        backend=backend,
    )
    model.fit(X, y, seed=train_seed)
    model.summary()
    posterior_y = model.predict(X, seed=predict_seed).mean(0)

    for (yy, pyy) in zip(y, posterior_y):
        print(yy, pyy)


app()
