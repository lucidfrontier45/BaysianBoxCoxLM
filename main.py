from typing import Optional

import numpy as np
import numpyro
import typer

from boxcox_lm import BackendType, BayesianBoxCoxLinearRegression

app = typer.Typer(name="baysian_boxcox_regression", add_completion=False)


@app.command()
def main(
    n_data: int = 50,
    w_s: float = 2.0,
    sigma_beta: float = 2.0,
    lambda_s: float = 1.0,
    n_warmup: int = 500,
    n_samples: int = 1000,
    n_chains: int = 4,
    seed: Optional[int] = None,
    backend: BackendType = typer.Option(BackendType.NumPyro, case_sensitive=False),
):

    numpyro.set_host_device_count(4)

    seed = 0
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
    model.fit(X, y)
    model.summary()
    posterior_y = model.predict(X).mean(0)

    for (yy, pyy) in zip(y, posterior_y):
        print(yy, pyy)


app()
