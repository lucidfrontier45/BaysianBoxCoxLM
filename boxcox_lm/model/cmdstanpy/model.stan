functions {
    /* ... function declarations and definitions ... */
    vector boxcox(vector y, real lambda, int N) {
        vector[N] z;
        real inv_lambda = 1.0 / lambda;
        for (i in 1:N){
            z[i] = (pow(y[i], lambda) - 1.0) * inv_lambda;
        }
        return z;
    }

    vector inv_boxcox(vector z, real lambda, int N) {
        vector[N] y;
        real inv_lambda = 1.0 / lambda;
        for (i in 1:N){
            y[i] = pow((z[i] * lambda + 1.0), inv_lambda);
        }
        return y;
    }
}

data {
    /* ... declarations ... */
    int N;
    int N_new;
    int D;
    matrix[N, D] X;
    vector[N] y;
    matrix[N_new, D] X_new;
    real<lower=0> w_s;
    real<lower=0> sigma_beta;
    real<lower=0> lambda_s;
}

parameters {
    /* ... declarations ... */
    vector[D] w;
    real<lower=0> sigma;
    real lambda;
}

transformed parameters {
    /* ... declarations ... statements ... */
    vector[N] z;
    z = (boxcox(y, lambda, N) - X * w ) / sigma;
}

model {
    /* ... declarations ... statements ... */
    w ~ normal(0.0, w_s);
    sigma ~ exponential(1.0 / sigma_beta);
    lambda ~ normal(1.0, lambda_s);
    z ~ normal(0.0, 1.0);
    target += (lambda - 1.0) * log(y);
    target += -N*log(sigma) ;
}

generated quantities {
    /* ... declarations ... statements ... */
    vector[N_new] yp;
    if (N_new > 0){
        yp = inv_boxcox(to_vector(normal_rng(X_new * w, sigma)), lambda, N_new);
    }
}