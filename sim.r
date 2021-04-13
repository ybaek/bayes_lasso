# Simulating data
library(genlasso)
library(rstan)
library(statmod)
#
set.seed(2021 - 4 - 13)
N <- 200
N_edges <- N - 1 # Like a Markov
beta_true <- numeric(N)
beta_true[1] <- rnorm(1)
block <- 10
for (i in 1:(N %/% block)) {
    inds <- seq(block * (i - 1) + 1, block * i)
    coin <- sample(2, 1, prob = c(.9, .1)) - 1
    if (!coin) beta_true[inds] <- rnorm(length(inds)) * .1
    else {
        sign <- (-1)^(sample(2, 1) - 1)
        beta_true[inds] <- rnorm(length(inds)) + sign * 10
    }
}
sigma <- abs(rnorm(1))
y <- rnorm(N) * sigma + beta_true
# Fused lasso solution paths (from genlasso package)
fused_paths <- fusedlasso1d(y = y, gamma = 1.0)
