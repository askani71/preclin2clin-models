# =============================================================================
# Gaussian Process Regression
#
# R port of python/gaussian_process.ipynb (same sections, same steps).
# Only base R + MASS (shipped with R) are used: no extra packages to install.
# The scikit-learn GaussianProcessRegressor section is reproduced with a small
# implementation of the same algorithm (zero-mean GP, kernel C * RBF, fixed
# noise alpha = sigma_n^2, hyperparameters fitted by maximizing the
# log-marginal likelihood with random restarts).
#
# Run interactively (e.g. RStudio) to see the plots one by one, or with
#   Rscript gaussian_process.R
# to save all plots in gaussian_process_plots.pdf (in the working directory).
# =============================================================================

# Fix the random seed so results are reproducible.
set.seed(42)

if (!interactive()) pdf("gaussian_process_plots.pdf", width = 16, height = 8)

colors <- c('#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c',
            '#137e6d', '#be0119', '#3b638c', '#af6f09', '#c79fef', '#80f9ad')
blues <- colorRampPalette(c('#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'))(100)

alpha_col <- function(col, alpha) adjustcolor(col, alpha.f = alpha)

plot_heatmap <- function(M, main) {
  # Draw M like seaborn.heatmap: row 1 at the top, colour bar on the right.
  op <- par(no.readonly = TRUE)
  on.exit({ par(op); layout(1) })
  layout(matrix(1:2, nrow = 1), widths = c(6, 1))
  par(mar = c(3, 3, 3, 1))
  image(seq_len(ncol(M)), seq_len(nrow(M)), t(M[nrow(M):1, , drop = FALSE]),
        col = blues, axes = FALSE, xlab = '', ylab = '', main = main, useRaster = TRUE)
  box()
  zr <- range(M)
  if (diff(zr) == 0) zr <- zr + c(-0.5, 0.5)
  z <- seq(zr[1], zr[2], length.out = 100)
  par(mar = c(3, 1, 3, 4))
  image(1, z, matrix(z, nrow = 1), col = blues, axes = FALSE, xlab = '', ylab = '')
  axis(4, las = 1)
  box()
}


# -----------------------------------------------------------------------------
# Hidden function
# f(x) = sin(3*pi*x/10) + cos(7*pi*x/10)
# -----------------------------------------------------------------------------
d <- 1

f <- function(x) sin((3*pi)*x/10) + cos((7*pi)*x/10)

xx <- seq(0, 10, length.out = 1000)
yy <- f(xx)


# -----------------------------------------------------------------------------
# Generate Data Samples
# -----------------------------------------------------------------------------
# Number of samples (training set).
# Note: the observed values are called t0 (not t) to avoid masking base::t().
n <- 10
x <- sort(runif(n) * 10)
t0 <- f(x)

plot(xx, yy, type = 'l', col = colors[2], lwd = 2, xlab = 'x', ylab = '',
     main = expression(paste('Graph of ', f(x) == sin(3*pi*x/10) + cos(7*pi*x/10),
                             ' with training set, no noise assumed')))
points(x, t0, pch = 19, col = colors[1])
legend('topleft', legend = c('f(x)', 'Training set, no noise'),
       col = colors[2:1], lty = c(1, NA), pch = c(NA, 19), lwd = 2)

# Next, we generate some training sample observations for the case when a
# gaussian noise is assumed.

# Error standard deviation.
sigma_n <- 0.2

# Errors.
epsilon <- rnorm(n, mean = 0, sd = sigma_n)

# Observed target variable.
t_n <- t0 + epsilon

# Let us see the error distribution.
hist(epsilon, freq = FALSE, col = alpha_col(colors[2], 0.4), border = 'white',
     main = 'Errors Distribution', xlab = '')
lines(density(epsilon), col = colors[2], lwd = 2)

# Let us now visualize the sample data.
plot(xx, yy, type = 'l', col = colors[2], lwd = 2, xlab = 'x', ylab = '',
     main = bquote(paste('Graph of ', f(x) == sin(3*pi*x/10) + cos(7*pi*x/10),
                         ' with training set, gaussian noise assumed, ', sigma[n] == .(sigma_n))))
points(x, t_n, pch = 19, col = colors[1])
legend('topleft', legend = c('f(x)', 'Training set, gaussian noise'),
       col = colors[2:1], lty = c(1, NA), pch = c(NA, 19), lwd = 2)

# We now consider test data points on which we want to generate predictions.
x_star <- xx


# -----------------------------------------------------------------------------
# Kernel Function
#
# A gaussian process is completely specified by its mean and covariance
# functions m(x), kappa(x, x'). Usually m(x) = 0, while a common choice for
# kappa is the squared exponential:
#   kappa(x_p, x_q) = sigma_f * exp(-|x_p - x_q|^2 / (2 * l^2))
# where sigma_f, l > 0 are hyperparameters.
# -----------------------------------------------------------------------------
kernel_function <- function(x, y, sigma_f = 1, l = 1) {
  # Squared exponential kernel, vectorized: returns the matrix kappa(x_i, y_j).
  sigma_f * exp(-outer(x, y, '-')^2 / (2 * l^2))
}

is_positive_definite <- function(A) {
  # Check positive definiteness via Cholesky (the determinant under/overflows on large matrices).
  !inherits(try(chol(A), silent = TRUE), 'try-error')
}

sample_mvn <- function(mean, cov, size = 1, jitter = 1e-8) {
  # Sample from N(mean, cov), adding a small jitter to the diagonal for numerical stability.
  # Returns a size x length(mean) matrix (one sample per row).
  cov <- (cov + t(cov)) / 2 + jitter * diag(nrow(cov))
  matrix(MASS::mvrnorm(size, mu = mean, Sigma = cov), nrow = size)
}

# Let us select the parameters.
l <- 0.8
sigma_f <- 1


# -----------------------------------------------------------------------------
# Compute Covariance Matrices
# K(X, X) in M_n, K(X_*, X) in M_{n_* x n}, K(X_*, X_*) in M_{n_*}.
# -----------------------------------------------------------------------------
compute_cov_matrices <- function(x, x_star, sigma_f = 1, l = 1, noise = TRUE, sigma_n = 0.1) {
  # Compute components of the covariance matrix of the joint distribution.
  #
  # We follow the notation:
  #   - K = K(X)
  #   - k_star = k(X_*, X)
  #   - K_star = K(X_*)
  n <- length(x)

  K <- kernel_function(x, x, sigma_f = sigma_f, l = l)
  K_star <- kernel_function(x_star, x_star, sigma_f = sigma_f, l = l)
  k_star <- kernel_function(x_star, x, sigma_f = sigma_f, l = l)

  if (noise) K <- K + (sigma_n^2) * diag(n)

  C <- rbind(cbind(K, t(k_star)),
             cbind(k_star, K_star))

  list(C = C, K = K, K_star = K_star, k_star = k_star)
}


# -----------------------------------------------------------------------------
# No noise assumption
#
# The observed values t are exactly f(X). Given the test points X*, the joint
# distribution of f and f* is N(0, C) with
#   C = | K(X)        k(X, X*) |
#       | k(X, X*)^T  K(X*)    |
# -----------------------------------------------------------------------------
# Let us compute the components.
cm <- compute_cov_matrices(x, x_star, sigma_f = sigma_f, l = l, noise = FALSE)
C_n <- cm$C; K_n <- cm$K; K_star_n <- cm$K_star; k_star <- cm$k_star

# K(X)
cat('K(X)\n')
cat(sprintf('Shape: (%d, %d)\n', nrow(K_n), ncol(K_n)))
cat(sprintf('Positive definite: %s\n', is_positive_definite(K_n)))
plot_heatmap(K_n, 'K(X), no noise')

# K(X_*) is positive semi-definite but numerically singular (1000 very close points):
# its smallest eigenvalue is ~0 (up to rounding), so Cholesky may fail.
cat('K(X_*)\n')
cat(sprintf('Shape: (%d, %d)\n', nrow(K_star_n), ncol(K_star_n)))
cat(sprintf('Positive definite: %s\n', is_positive_definite(K_star_n)))
cat(sprintf('Min eigenvalue: %.2e\n', min(eigen(K_star_n, symmetric = TRUE, only.values = TRUE)$values)))
plot_heatmap(K_star_n, 'K(X_*), no noise')

# k(X_*, X)
cat('k(X_*, X)\n')
cat(sprintf('Shape: (%d, %d)\n', nrow(k_star), ncol(k_star)))
plot_heatmap(k_star, 'k(X_*, X)')

# The covariance matrix C_n.
cat('C_n\n')
cat(sprintf('Shape: (%d, %d)\n', nrow(C_n), ncol(C_n)))
cat(sprintf('Positive definite: %s\n', is_positive_definite(C_n)))
cat(sprintf('Min eigenvalue: %.2e\n', min(eigen(C_n, symmetric = TRUE, only.values = TRUE)$values)))
plot_heatmap(C_n, 'Covariance Matrix C_n, no noise')

# The prior distribution for f* is N(mu, Sigma), with mu = 0 and Sigma = K(X*).
mean_z <- rep(0, length(xx))
# Sample from prior distribution.
z_star <- sample_mvn(mean = mean_z, cov = K_star_n, size = 20)
matplot(xx, t(z_star), type = 'l', lty = 1, col = alpha_col(colors[1], 0.3),
        xlab = 'x', ylab = '', main = 'Samples of Prior Distribution, no noise')
lines(xx, yy, col = colors[2], lwd = 2)
legend('bottomright', legend = 'f(x)', col = colors[2], lwd = 2)

compute_gpr_parameters <- function(K, K_star, k_star, y) {
  # Compute gaussian regression parameters.
  # Solve linear systems instead of inverting K (more stable and faster);
  # tol = 0 disables R's conditioning check, as numpy does.
  # Mean.
  f_bar_star <- k_star %*% solve(K, y, tol = 0)
  # Covariance.
  cov_f_star <- K_star - k_star %*% solve(K, t(k_star), tol = 0)

  list(mean = as.vector(f_bar_star), cov = cov_f_star)
}

gpr <- compute_gpr_parameters(K_n, K_star_n, k_star, t0)
m <- gpr$mean; c_ <- gpr$cov

# Sample from joint distribution.
z_star <- sample_mvn(mean = m, cov = c_, size = 20)
matplot(xx, t(z_star), type = 'l', lty = 1, col = alpha_col(colors[1], 0.3),
        xlab = 'x', ylab = '', main = 'Samples of Posterior Distribution, no noise')
points(x, t0, pch = 19, col = colors[8])
# Plot "true" function.
lines(xx, yy, col = colors[2], lwd = 2)
legend('bottomright', legend = 'f(x)', col = colors[2], lwd = 2)

predict_gp <- function(x_new, X_train, t_train, noise = FALSE) {
  # Predictive mean and standard deviation at the points x_new (vectorized).
  # Uses the global hyperparameters l, sigma_f, sigma_n.
  cm <- compute_cov_matrices(X_train, x_new, l = l, sigma_n = sigma_n, sigma_f = sigma_f, noise = noise)
  gpr <- compute_gpr_parameters(cm$K, cm$K_star, cm$k_star, t_train)
  # Clip tiny negative variances caused by rounding before taking the square root.
  list(mean = gpr$mean, sd = sqrt(pmax(diag(gpr$cov), 0)))
}

plot_predictive <- function(pred, t_train, main) {
  plot(xx, pred$mean, type = 'n', ylim = range(pred$mean - 2*pred$sd, pred$mean + 2*pred$sd, yy),
       xlab = 'x', ylab = '', main = main)
  polygon(c(xx, rev(xx)), c(pred$mean - 2*pred$sd, rev(pred$mean + 2*pred$sd)),
          col = alpha_col(colors[4], 0.3), border = NA)
  lines(xx, pred$mean, col = colors[4], lwd = 2)
  points(x, t_train, pch = 19, col = colors[5])
  # Plot "true" function.
  lines(xx, yy, col = colors[2], lwd = 2)
  legend('bottomright', legend = c('mean prediction', '±2 std', 'f(x)'),
         col = c(colors[4], alpha_col(colors[4], 0.3), colors[2]),
         lty = c(1, NA, 1), pch = c(NA, 15, NA), pt.cex = 2, lwd = 2)
}

plot_predictive(predict_gp(xx, x, t0), t0, 'Predictive Distribution, no noise')


# -----------------------------------------------------------------------------
# Gaussian noise assumption
#
# The joint distribution of y and f* is N(0, C) with
#   C = | K(X, X) + sigma_n^2 I   K(X, X*) |
#       | K(X*, X)                K(X*, X*) |
# The sigma_n^2 I term accounts for additive i.i.d. gaussian noise.
# -----------------------------------------------------------------------------
cm <- compute_cov_matrices(x, x_star, sigma_f = sigma_f, sigma_n = sigma_n, l = l, noise = TRUE)
C_g <- cm$C; K_g <- cm$K; K_star_g <- cm$K_star; k_star <- cm$k_star

plot_heatmap(K_g, 'K(X), gaussian noise')
plot_heatmap(K_star_g, 'K(X_*), gaussian noise')
plot_heatmap(C_g, 'Covariance Matrix C_g, gaussian noise')

# Let us verify that C is symmetric.
cat(sprintf('C_g symmetric: %s\n', isTRUE(all(t(C_g) == C_g))))

# Verify C is positive definite.
# The noise term only regularizes K(X); the K(X_*) block stays numerically singular.
cat(sprintf('Positive definite: %s\n', is_positive_definite(C_g)))
cat(sprintf('Min eigenvalue: %.2e\n', min(eigen(C_g, symmetric = TRUE, only.values = TRUE)$values)))


# -----------------------------------------------------------------------------
# Prior Distribution
# -----------------------------------------------------------------------------
mean_z <- rep(0, length(xx))
# Sample from prior distribution.
z_star <- sample_mvn(mean = mean_z, cov = K_star_g, size = 20)
matplot(xx, t(z_star), type = 'l', lty = 1, col = alpha_col(colors[1], 0.3),
        xlab = 'x', ylab = '', main = 'Samples of Prior Distribution, gaussian noise')
lines(xx, yy, col = colors[2], lwd = 2)
legend('bottomright', legend = 'f(x)', col = colors[2], lwd = 2)

# Use the noisy observations t_n (not the noise-free t0).
gpr <- compute_gpr_parameters(K_g, K_star_g, k_star, t_n)
m_g <- gpr$mean; c_g <- gpr$cov

# Sample from joint distribution.
z_star <- sample_mvn(mean = m_g, cov = c_g, size = 20)
matplot(xx, t(z_star), type = 'l', lty = 1, col = alpha_col(colors[1], 0.2),
        xlab = 'x', ylab = '', main = 'Samples of Posterior Distribution, gaussian noise')
points(x, t_n, pch = 19, col = colors[8])
# Plot "true" function.
lines(xx, yy, col = colors[2], lwd = 2)
legend('bottomright', legend = 'f(x)', col = colors[2], lwd = 2)

plot_predictive(predict_gp(xx, x, t_n, noise = TRUE), t_n, 'Predictive Distribution, gaussian noise')


# -----------------------------------------------------------------------------
# Hyperparameters
#
# - sigma_f describes the amplitude of the function.
# - l is a locality parameter, i.e. how far the points interact.
# -----------------------------------------------------------------------------
plot_posterior_samples <- function(f_bar_star, cov_f_star) {
  z_star <- sample_mvn(mean = f_bar_star, cov = cov_f_star, size = 20)
  matplot(xx, t(z_star), type = 'l', lty = 1, col = alpha_col(colors[1], 0.2), xlab = 'x', ylab = '',
          main = bquote(paste('Samples of Posterior Distribution, ',
                              sigma[f] == .(sigma_f), ' and ', italic(l) == .(l))))
  lines(xx, yy, col = colors[2], lwd = 2)
  legend('topright', legend = 'f(x)', col = colors[2], lwd = 2)
}

# - sigma_f = 2 and l = 1
l <- 1
sigma_f <- 2

# Compute covariance matrices.
cm <- compute_cov_matrices(x, x_star, sigma_f = sigma_f, l = l, noise = FALSE)
# Compute gaussian process parameters.
gpr <- compute_gpr_parameters(cm$K, cm$K_star, cm$k_star, t0)
# Plot posterior covariance matrix components.
plot_heatmap(gpr$cov, sprintf('Components of the Covariance Matrix cov_f_star, sigma_f = %s and l = %s', sigma_f, l))

# In this case the values of the posterior covariance matrix are not that
# localized: points far away can still interact, i.e. the fit becomes more
# global. Let us plot the resulting fit.
cm <- compute_cov_matrices(x, x_star, sigma_f = sigma_f, sigma_n = sigma_n, l = l, noise = TRUE)
gpr <- compute_gpr_parameters(cm$K, cm$K_star, cm$k_star, t_n)
plot_posterior_samples(gpr$mean, gpr$cov)

# - sigma_f = 2 and l = 0.001
l <- 0.001
sigma_f <- 2

# Compute covariance matrices.
cm <- compute_cov_matrices(x, x_star, sigma_f = sigma_f, sigma_n = sigma_n, l = l, noise = TRUE)
# Compute gaussian process parameters.
gpr <- compute_gpr_parameters(cm$K, cm$K_star, cm$k_star, t_n)
# Plot posterior covariance matrix components.
plot_heatmap(gpr$cov, sprintf('Components of the Covariance Matrix cov_f_star, sigma_f = %s and l = %s', sigma_f, l))

# In contrast, the higher values of the posterior covariance matrix are now
# concentrated along the diagonal: points far away have no effect on each
# other, i.e. the fit becomes more local.
plot_posterior_samples(gpr$mean, gpr$cov)

# Hence, the hyperparameter l somehow encodes the "complexity" and "locality"
# of the model.

# - sigma_f = 50 and l = 0.1
l <- 0.1
sigma_f <- 50

# Compute covariance matrices.
cm <- compute_cov_matrices(x, x_star, sigma_f = sigma_f, sigma_n = sigma_n, l = l, noise = TRUE)
# Compute gaussian process parameters.
gpr <- compute_gpr_parameters(cm$K, cm$K_star, cm$k_star, t_n)
# Plot posterior covariance matrix components.
plot_heatmap(gpr$cov, sprintf('Components of the Covariance Matrix cov_f_star, sigma_f = %s and l = %s', sigma_f, l))
plot_posterior_samples(gpr$mean, gpr$cov)

# The hyperparameter sigma_f encodes the amplitude of the fit.


# -----------------------------------------------------------------------------
# GaussianProcessRegressor (equivalent of the scikit-learn section)
#
# Kernel: ConstantKernel(sigma_f) * RBF(l), i.e. kernel_function() above.
# The bounds on each kernel component are the lower and upper bounds of the
# hyperparameters. The fit selects the hyperparameters that maximize the
# log-marginal likelihood (the integral of the likelihood times the prior),
# with n_restarts random restarts as in scikit-learn.
# -----------------------------------------------------------------------------
gp_regressor <- function(sigma_f, l, sigma_f_bounds, l_bounds, alpha, n_restarts = 10) {
  list(sigma_f = sigma_f, l = l, sigma_f_bounds = sigma_f_bounds, l_bounds = l_bounds,
       alpha = alpha, n_restarts = n_restarts)
}

gp_fit <- function(gp, X, y) {
  # Fit to data using Maximum Likelihood Estimation of the kernel hyperparameters.
  n <- length(X)
  D2 <- outer(X, X, '-')^2
  neg_log_marginal_likelihood <- function(log_theta) {
    K <- kernel_function(X, X, sigma_f = exp(log_theta[1]), l = exp(log_theta[2])) + gp$alpha * diag(n)
    L <- tryCatch(chol(K), error = function(e) NULL)
    if (is.null(L)) return(1e25)
    a <- backsolve(L, forwardsolve(t(L), y))
    0.5 * sum(y * a) + sum(log(diag(L))) + 0.5 * n * log(2 * pi)
  }
  # Analytic gradient w.r.t. (log sigma_f, log l), as scikit-learn uses:
  # d/dtheta_j = 0.5 * tr((K^-1 - a a^T) dK/dtheta_j).
  neg_log_marginal_likelihood_grad <- function(log_theta) {
    K_f <- kernel_function(X, X, sigma_f = exp(log_theta[1]), l = exp(log_theta[2]))
    L <- tryCatch(chol(K_f + gp$alpha * diag(n)), error = function(e) NULL)
    if (is.null(L)) return(c(0, 0))
    a <- backsolve(L, forwardsolve(t(L), y))
    W <- chol2inv(L) - tcrossprod(a)
    dK_dlog_sigma_f <- K_f
    dK_dlog_l <- K_f * D2 / exp(2 * log_theta[2])
    0.5 * c(sum(W * dK_dlog_sigma_f), sum(W * dK_dlog_l))
  }
  # Optimize in log-space within the bounds: first from the initial values,
  # then from n_restarts starting points drawn log-uniformly within the bounds.
  lower <- log(c(gp$sigma_f_bounds[1], gp$l_bounds[1]))
  upper <- log(c(gp$sigma_f_bounds[2], gp$l_bounds[2]))
  starts <- rbind(log(c(gp$sigma_f, gp$l)),
                  cbind(runif(gp$n_restarts, lower[1], upper[1]),
                        runif(gp$n_restarts, lower[2], upper[2])))
  fits <- lapply(seq_len(nrow(starts)), function(i)
    optim(starts[i, ], neg_log_marginal_likelihood, neg_log_marginal_likelihood_grad,
          method = 'L-BFGS-B', lower = lower, upper = upper))
  best <- fits[[which.min(vapply(fits, function(o) o$value, numeric(1)))]]

  gp$sigma_f_ <- exp(best$par[1])
  gp$l_ <- exp(best$par[2])
  gp$log_marginal_likelihood <- -best$value
  gp$X_train <- X
  gp$L <- chol(kernel_function(X, X, sigma_f = gp$sigma_f_, l = gp$l_) + gp$alpha * diag(n))
  gp$alpha_vec <- backsolve(gp$L, forwardsolve(t(gp$L), y))
  gp
}

gp_kernel_label <- function(gp) {
  # Same format as scikit-learn's kernel_ repr, e.g. "1.41**2 * RBF(length_scale=0.1)".
  sprintf('%.3g**2 * RBF(length_scale=%.3g)', sqrt(gp$sigma_f_), gp$l_)
}

gp_predict <- function(gp, X_new, return_cov = FALSE) {
  # Posterior mean, standard deviation and (optionally) covariance at X_new.
  k_star <- kernel_function(X_new, gp$X_train, sigma_f = gp$sigma_f_, l = gp$l_)
  mean <- as.vector(k_star %*% gp$alpha_vec)
  v <- forwardsolve(t(gp$L), t(k_star))
  if (return_cov) {
    cov <- kernel_function(X_new, X_new, sigma_f = gp$sigma_f_, l = gp$l_) - crossprod(v)
    return(list(mean = mean, sd = sqrt(pmax(diag(cov), 0)), cov = cov))
  }
  list(mean = mean, sd = sqrt(pmax(gp$sigma_f_ - colSums(v^2), 0)))
}

gp_sample_y <- function(gp, X_new, n_samples = 1) {
  # Samples from the posterior: a length(X_new) x n_samples matrix (as scikit-learn's sample_y).
  pred <- gp_predict(gp, X_new, return_cov = TRUE)
  t(sample_mvn(mean = pred$mean, cov = pred$cov, size = n_samples))
}

# Define kernel parameters.
l <- 0.1
sigma_f <- 2

# Define GaussianProcessRegressor object.
gp <- gp_regressor(sigma_f = sigma_f, l = l, sigma_f_bounds = c(1e-3, 1e3), l_bounds = c(1e-3, 1e3),
                   alpha = sigma_n^2, n_restarts = 10)

cat(sprintf('xx length: %d\n', length(xx)))

# Fit to the noisy observations (alpha = sigma_n^2 models the noise),
# estimating the kernel hyperparameters by maximum likelihood.
gp <- gp_fit(gp, x, t_n)
cat(sprintf('Optimized kernel: %s\n', gp_kernel_label(gp)))

# Predictions: make the prediction on test set.
y_pred <- gp_predict(gp, xx)$mean
cat(sprintf('y_pred length: %d\n', length(y_pred)))

# Let us plot the prediction.
# The title shows the optimized kernel (sigma_f and l above are only the starting values).
plot(xx, yy, type = 'l', col = colors[1], lwd = 2, xlab = 'x', ylab = '',
     ylim = range(yy, y_pred), main = sprintf('Prediction GaussianProcessRegressor, kernel = %s', gp_kernel_label(gp)))
lines(xx, y_pred, col = colors[2], lwd = 2)
legend('topright', legend = c('f(x)', 'pred'), col = colors[1:2], lwd = 2)

# Credible Interval.
# Generate samples from the posterior distribution of the fitted model.
y_hat_samples <- gp_sample_y(gp, xx, n_samples = 100)
# Compute the mean of the sample.
y_hat <- rowMeans(y_hat_samples)
# Compute the standard deviation of the sample (population sd, as numpy's std).
y_hat_sd <- sqrt(rowMeans((y_hat_samples - y_hat)^2))

plot_credible_interval <- function(x_train, y_train, x_grid, f_grid, y_hat, y_hat_sd, y_pred, main, legend_pos) {
  lower <- y_hat - 2*y_hat_sd
  upper <- y_hat + 2*y_hat_sd
  plot(x_train, y_train, pch = 19, col = colors[1], xlim = range(x_grid),
       ylim = range(y_train, lower, upper, f_grid), xlab = 'x', ylab = '', main = main)
  # Plot corridor.
  polygon(c(x_grid, rev(x_grid)), c(lower, rev(upper)), col = adjustcolor('green', 0.3), border = NA)
  # Plot "true" function.
  lines(x_grid, f_grid, col = 'red', lwd = 2)
  # Plot prediction.
  lines(x_grid, y_pred, col = 'darkgreen', lwd = 2)
  legend(legend_pos, legend = c('training data', 'Credible Interval', 'f(x)', 'pred'),
         col = c(colors[1], adjustcolor('green', 0.3), 'red', 'darkgreen'),
         pch = c(19, 15, NA, NA), lty = c(NA, NA, 1, 1), pt.cex = c(1, 2, 1, 1), lwd = 2)
}

plot_credible_interval(x, t_n, xx, yy, y_hat, y_hat_sd, y_pred,
                       sprintf('Prediction & Credible Interval, kernel = %s', gp_kernel_label(gp)), 'topright')


# -----------------------------------------------------------------------------
# Summary & Final Example
#
# A self-contained example using only the GaussianProcessRegressor equivalent.
# -----------------------------------------------------------------------------

# Construct Training Set.
# Set dimension.
d <- 1
# Number of training points.
n <- 1000
# Length of the training set.
L <- 2
# Generate training features.
x <- seq(0, L, length.out = n)
# Error standard deviation.
sigma_n <- 0.4
# Errors.
epsilon <- rnorm(n, mean = 0, sd = sigma_n)
# Generate non-linear function.
f <- function(x) sin((4*pi)*x) + sin((7*pi)*x) + sin((3*pi)*x)

f_x <- f(x)

# Observed target variable.
y <- f_x + epsilon

# Construct Test Set.
n_star <- n + 300
x_star <- seq(0, L + 0.5, length.out = n_star)

# Define Kernel and Model.
# Define kernel parameters.
l <- 0.1
sigma_f <- 2

# Define GaussianProcessRegressor object.
gp <- gp_regressor(sigma_f = sigma_f, l = l, sigma_f_bounds = c(1e-2, 1e2), l_bounds = c(1e-2, 1e2),
                   alpha = sigma_n^2, n_restarts = 10)

# Model Fit + Predictions.
# Fit to data using Maximum Likelihood Estimation of the parameters.
gp <- gp_fit(gp, x, y)
cat(sprintf('Optimized kernel (final example): %s\n', gp_kernel_label(gp)))
# Make the prediction on test set.
y_pred <- gp_predict(gp, x_star)$mean

# Generate samples from posterior distribution.
y_hat_samples <- gp_sample_y(gp, x_star, n_samples = 100)
# Compute the mean of the sample.
y_hat <- rowMeans(y_hat_samples)
# Compute the standard deviation of the sample.
y_hat_sd <- sqrt(rowMeans((y_hat_samples - y_hat)^2))

# Visualize Results.
plot_credible_interval(x, y, x_star, f(x_star), y_hat, y_hat_sd, y_pred,
                       'Prediction & Credible Interval', 'bottomleft')

if (!interactive()) invisible(dev.off())
