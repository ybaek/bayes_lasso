data {
    int<lower=1> N;
    int<lower=0> N_edges;
    int<lower=1, upper=N> node1[N_edges];
    int<lower=1, upper=N> node2[N_edges];
    vector[N] y;
}
transformed data {
    vector[N] zeros = rep_vector(0, N);
    matrix[N_edges, N] D; // "penalty" matrix
    for (i in 1:N_edges) {
        D[i, node1[i]] = -1;
        D[i, node2[i]] = 1;
    } 
}
parameters {
    vector[N] beta; // coefficient over a graph
    vector<lower=0>[N] tau_v; // vertex shrinkage
    vector<lower=0>[N_edges] tau_e; // edge shrinkage
    real<lower=0> sigma;
}
transformed parameters {
    // Explicitly forming a coefficient precision matrix
    matrix[N,N] pseudo_lap = D' * diag_pre_multiply(1. ./ tau_e, D);
    matrix[N,N] beta_prec = add_diag(pseudo_lap, 1. ./ tau_v);
}
model {
    sigma ~ inv_chi_square(1);
    tau_v ~ exponential(1.0);   
    tau_e ~ exponential(1.0);
    beta ~ multi_normal_prec(zeros, beta_prec);
    y ~ normal(beta, pow(sigma, -0.5));
}
