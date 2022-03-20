import numpy as np
import numpyro

from boxcox_lm import BackendType, BayesianBoxCoxLinearRegression

numpyro.set_host_device_count(4)

seed = 0
np.random.seed(seed=seed)
N = 20
D = 2

w = np.array([5.0, -1.2])
X = np.random.randn(N, D - 1) * 0.1
X = np.column_stack([np.ones(N), X])
z = np.dot(X, w)
y = np.array([np.random.poisson(zz) for zz in np.exp(z)])

params = {
    "w_s": 2,
    "sigma_beta": 2,
    "lambda_s": 1,
    "n_warmup": 500,
    "n_samples": 1000,
    "n_chains": 4,
}


model = BayesianBoxCoxLinearRegression(backend=BackendType.CmdStanPy, **params)
model.fit(X, y)
model.summary()
posterior_y = model.predict(X).mean(0)

for (yy, pyy) in zip(y, posterior_y):
    print(yy, pyy)
