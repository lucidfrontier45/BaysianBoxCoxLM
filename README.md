# Fully Bayesian Inference for Linear Regression Model with Box-Cox Transform

This repository includes CmdStanPy and NumPyro implementations of the Linear Regression Model with Box-Cox Transform.

# Install

```sh
poetry install

# without development dependencies
poetry install --no-dev
```

# Run

```
$ python main.py --help
Usage: main.py [OPTIONS]

Options:
  --n-data INTEGER               number of input data  [default: 50]
  --w-s FLOAT                    parameter of the w prior  [default: 2.0]
  --sigma-beta FLOAT             parameter of the sigma prior  [default: 2.0]
  --lambda-s FLOAT               parameter of the lambda prior  [default: 2.0]
  --n-warmup INTEGER             MCMC warmup samples  [default: 500]
  --n-samples INTEGER            MCMC samples  [default: 1000]
  --n-chains INTEGER             MCMC chains  [default: 4]
  --seed INTEGER
  --backend [numpyro|cmdstanpy]  [default: BackendType.NumPyro]
  --help                         Show this message and exit.
```