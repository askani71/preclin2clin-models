# =============================================================================
# Elastic Net Regression
#
# R port of python/elastic_net.ipynb (same sections, same steps).
# Only base R is used: no extra packages to install.
# The scikit-learn sections (ElasticNet, ElasticNetCV, LassoCV, RidgeCV) are
# reproduced with the coordinate descent implemented below and a small k-fold
# cross-validation. The random numbers differ from numpy, so the numbers are
# close to (not equal to) the ones in the Python notebook.
#
# Run interactively (e.g. RStudio) to see the plots one by one, or with
#   Rscript elastic_net.R
# to save all plots in elastic_net_plots.pdf (in the working directory).
# =============================================================================

# Fix the random seed so results are reproducible.
set.seed(42)

if (!interactive()) pdf("elastic_net_plots.pdf", width = 16, height = 8)

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

plot_coefficients <- function(w, main, color = colors[1]) {
  # Bar chart of the weights w_1, ..., w_p.
  barplot(w, names.arg = seq_along(w), col = color, border = NA,
          xlab = 'feature j', ylab = expression(w[j]), main = main)
  abline(h = 0)
}

plot_fit <- function(pred, y_train, main, pred_label = 'prediction') {
  # Prediction on the test points, training data and "true" function.
  plot(xx, yy, type = 'n', ylim = c(-3, 3), xlab = 'x', ylab = '', main = main)
  lines(xx, pred, col = colors[4], lwd = 2)
  points(x, y_train, pch = 19, col = colors[5])
  lines(xx, yy, col = colors[2], lwd = 2)
  legend('bottomright', legend = c(pred_label, 'f(x)'), col = colors[c(4, 2)], lwd = 2)
}


# -----------------------------------------------------------------------------
# Hidden function
# f(x) = sin(3*pi*x/10) + cos(7*pi*x/10)
# (the same function used in the Gaussian process tutorial).
# -----------------------------------------------------------------------------
d <- 1

f <- function(x) sin((3*pi)*x/10) + cos((7*pi)*x/10)

xx <- seq(0, 10, length.out = 1000)
yy <- f(xx)


# -----------------------------------------------------------------------------
# Generate Data Samples
# -----------------------------------------------------------------------------
# Number of samples (training set).
# A linear model with many features needs more points than the 10 used for the GP.
# Note: the observed values are called t0 (not t) to avoid masking base::t().
n <- 20
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
# Feature Map
#
# Elastic Net is a linear model. To fit the non-linear f we map each input x
# to the gaussian radial basis functions (RBF)
#   phi_j(x) = exp(-(x - c_j)^2 / (2 * l^2)),  j = 1, ..., p
# centered at p equally spaced points c_j in [0, 10], and consider the model
#   t = b + phi(x)^T w + epsilon
# (non-linear in x, linear in the intercept b and the weights w).
# The intercept is not penalized: each feature is centered on the training set,
# so the intercept is the mean of the targets. RBF features all lie in [0, 1],
# so they are not rescaled.
# -----------------------------------------------------------------------------
feature_map <- function(x, p = 17, l = 1, x_min = 0, x_max = 10) {
  # Gaussian RBF features, p centers equally spaced in [x_min, x_max]: returns Phi(x).
  exp(-outer(x, seq(x_min, x_max, length.out = p), '-')^2 / (2 * l^2))
}

center <- function(Phi, mean = colMeans(Phi)) {
  # Center the columns of Phi (with the training mean when given).
  list(Z = sweep(Phi, 2, mean), mean = mean)
}

is_positive_definite <- function(A) {
  # Check positive definiteness via Cholesky (the determinant under/overflows on large matrices).
  !inherits(try(chol(A), silent = TRUE), 'try-error')
}

# Let us select the parameters.
p <- 17
l <- 1


# -----------------------------------------------------------------------------
# Compute Design Matrices
# Phi(X) in M_{n x p}, Phi(X_*) in M_{n_* x p}; Z and Z_* are their centered
# versions (both use the mean of the training set).
# -----------------------------------------------------------------------------
Phi <- feature_map(x, p = p, l = l)
Phi_star <- feature_map(x_star, p = p, l = l)

centered <- center(Phi)
Z <- centered$Z
Phi_mean <- centered$mean
Z_star <- center(Phi_star, Phi_mean)$Z

# Phi(X)
cat(sprintf('Shape Phi(X): (%d, %d)\n', nrow(Phi), ncol(Phi)))
plot_heatmap(Phi, expression(paste('Design matrix ', Phi(X))))

# Phi(X_*)
cat(sprintf('Shape Phi(X_*): (%d, %d)\n', nrow(Phi_star), ncol(Phi_star)))

# Let us plot the features phi_j(x) on the test points (the columns of Phi(X_*)).
matplot(xx, Phi_star, type = 'l', lty = 1, lwd = 2, col = rep(colors, length.out = p),
        xlab = 'x', ylab = '', main = bquote(paste('RBF features, p = ', .(p), ' and ', l == .(l))))
points(x, rep(0, n), pch = '|', cex = 2)

# Z^T Z / n is the (sample) covariance matrix of the features. Neighbouring
# RBFs overlap, so their features are strongly correlated (multicollinearity).
G <- crossprod(Z) / n
plot_heatmap(cor(Z), 'Feature correlation matrix')

# Z^T Z is positive definite but nearly singular: its smallest eigenvalue is
# almost zero and its condition number is huge.
eig_G <- eigen(G, symmetric = TRUE, only.values = TRUE)$values
cat(sprintf('Positive definite: %s\n', is_positive_definite(G)))
cat(sprintf('Min eigenvalue: %.2e\n', min(eig_G)))
cat(sprintf('Condition number of Z^T Z/n: %.2e\n', max(eig_G) / min(eig_G)))


# -----------------------------------------------------------------------------
# Ordinary Least Squares
# w_OLS = argmin 1/(2n) |y - mean(y) - Z w|^2,  b = mean(y)
# Predictions on the test points: f_* = b + Z_* w.
# -----------------------------------------------------------------------------
fit_ols <- function(Z, y) {
  # Least squares weights and intercept (QR with a small tolerance instead of inverting Z^T Z).
  b <- mean(y)
  list(w = qr.coef(qr(Z, tol = 1e-12), y - b), b = b)
}

# Let us first fit the noise-free observations t0.
ols <- fit_ols(Z, t0)
plot_fit(ols$b + Z_star %*% ols$w, t0, 'OLS fit, no noise', 'OLS prediction')

# And now the noisy observations t_n.
ols_n <- fit_ols(Z, t_n)
plot_fit(ols_n$b + Z_star %*% ols_n$w, t_n, 'OLS fit, gaussian noise', 'OLS prediction')

# With noise, OLS overfits: the weights are huge and alternate in sign, because
# highly correlated features cancel each other out.
plot_coefficients(ols_n$w, 'OLS weights, gaussian noise')
cat(sprintf('Max |w_j|: %.1f\n', max(abs(ols_n$w))))


# -----------------------------------------------------------------------------
# Elastic Net Objective
#
#   w = argmin 1/(2n) |y - mean(y) - Z w|^2 + alpha * rho * |w|_1
#                                           + alpha * (1 - rho) / 2 * |w|_2^2
#
# alpha >= 0 is the strength of the penalty (alpha = 0 is OLS) and
# rho in [0, 1] (l1_ratio) the mix: rho = 0 is Ridge, rho = 1 is the Lasso.
#
# Coordinate Descent: minimize in one weight at a time; with z_j the j-th
# column of Z,
#   w_j <- S(z_j^T r_(j) / n, alpha * rho) / (|z_j|^2 / n + alpha * (1 - rho))
# where r_(j) is the residual without feature j and S the soft-thresholding
#   S(z, gamma) = sign(z) * max(|z| - gamma, 0).
# -----------------------------------------------------------------------------
soft_threshold <- function(z, gamma) sign(z) * pmax(abs(z) - gamma, 0)

elastic_net_cd <- function(Z, y, alpha, l1_ratio = 0.5, w_init = NULL, max_iter = 100000, tol = 1e-8) {
  # Elastic Net by cyclic coordinate descent (scikit-learn objective, centered Z).
  # Returns the weights w and the intercept b = mean(y).
  n <- nrow(Z)
  p <- ncol(Z)
  b <- mean(y)
  w <- if (is.null(w_init)) numeric(p) else w_init
  # Residual r = y - b - Z w, updated when a weight changes.
  r <- as.vector(y - b - Z %*% w)
  col_norm <- colSums(Z^2) / n
  # Active set (to make the R loops faster, same solution as the Python
  # version): after a sweep over all the weights, cycle only over the non-zero
  # ones until they converge, then check with a new full sweep.
  full <- TRUE
  for (iter in seq_len(max_iter)) {
    max_change <- 0
    for (j in if (full) seq_len(p) else which(w != 0)) {
      w_j <- w[j]
      # z_j^T r_(j) / n, where r_(j) adds back the contribution of feature j.
      rho_j <- sum(Z[, j] * r) / n + col_norm[j] * w_j
      w[j] <- soft_threshold(rho_j, alpha * l1_ratio) / (col_norm[j] + alpha * (1 - l1_ratio))
      if (w[j] != w_j) {
        r <- r - Z[, j] * (w[j] - w_j)
        max_change <- max(max_change, abs(w[j] - w_j))
      }
    }
    if (max_change < tol) {
      if (full) break
      full <- TRUE
    } else {
      full <- FALSE
    }
  }
  list(w = w, b = b)
}

# Ridge as a special case: for rho = 0 the solution has the closed form
#   w = (Z^T Z / n + alpha I)^{-1} Z^T (y - mean(y)) / n.
ridge_closed_form <- function(Z, y, alpha) {
  # Ridge weights (scikit-learn ElasticNet objective with l1_ratio = 0).
  n <- nrow(Z)
  b <- mean(y)
  list(w = solve(crossprod(Z) / n + alpha * diag(ncol(Z)), crossprod(Z, y - b) / n)[, 1], b = b)
}

# Let us verify that coordinate descent with rho = 0 recovers the closed form.
alpha <- 0.001
w_cd <- elastic_net_cd(Z, t_n, alpha = alpha, l1_ratio = 0)$w
w_ridge <- ridge_closed_form(Z, t_n, alpha = alpha)$w
cat(sprintf('CD (rho = 0) equals the Ridge closed form: %s\n', isTRUE(all.equal(w_cd, w_ridge, tolerance = 1e-6))))

# Adding alpha I lifts all the eigenvalues by alpha: this cures the ill-conditioning.
cat(sprintf('Condition number of Z^T Z/n + alpha I: %.2e\n', (max(eig_G) + alpha) / (min(eig_G) + alpha)))


# -----------------------------------------------------------------------------
# Regularization Path
# For fixed rho, the curve alpha -> w(alpha), computed from the largest alpha
# to the smallest one using each solution as the starting point of the next
# (warm start).
# -----------------------------------------------------------------------------
l1_ratio <- 0.5
alphas <- 10^seq(0, -6, length.out = 50)

coef_path <- matrix(0, nrow = length(alphas), ncol = p)
w <- NULL
for (i in seq_along(alphas)) {
  # max_iter is lower than in Python: R loops are slower, and with warm starts
  # the path is already close to convergence.
  w <- elastic_net_cd(Z, t_n, alpha = alphas[i], l1_ratio = l1_ratio, w_init = w, tol = 1e-6, max_iter = 5000)$w
  coef_path[i, ] <- w
}

matplot(alphas, coef_path, type = 'l', lty = 1, lwd = 2, col = rep(colors, length.out = p),
        log = 'x', xlim = rev(range(alphas)), xlab = expression(alpha), ylab = 'weight',
        main = bquote(paste('Elastic Net regularization path, ', rho, ' = ', .(l1_ratio))))
legend('topleft', legend = paste0('w', seq_len(p)), col = rep(colors, length.out = p), lwd = 2, ncol = 2)

# Let us visualize the fitted functions along the path, from the strongest
# penalty (light) to the weakest one (dark).
path_idx <- seq(1, length(alphas), by = 5)
path_cols <- colorRampPalette(c('#c6dbef', '#08306b'))(length(path_idx))
plot(xx, yy, type = 'n', ylim = c(-3, 3), xlab = 'x', ylab = '',
     main = expression(paste('Elastic Net fits along the regularization path (', alpha, ' from 1 to ', 10^-6, ')')))
for (k in seq_along(path_idx)) {
  lines(xx, mean(t_n) + Z_star %*% coef_path[path_idx[k], ], col = path_cols[k], lwd = 2)
}
points(x, t_n, pch = 19, col = colors[8])
lines(xx, yy, col = colors[2], lwd = 2)
legend('bottomright', legend = 'f(x)', col = colors[2], lwd = 2)


# -----------------------------------------------------------------------------
# Hyperparameters
# - alpha describes the strength of the regularization (bias-variance
#   trade-off): small alpha gives a flexible model, large alpha a rigid one.
# - rho describes the type of regularization: how many weights are set exactly
#   to zero (rho -> 1) versus shrunk smoothly (rho -> 0).
# -----------------------------------------------------------------------------
fit_and_plot <- function(alpha, l1_ratio) {
  # Fit the Elastic Net by coordinate descent, plot the weights and the fit.
  fit <- elastic_net_cd(Z, t_n, alpha = alpha, l1_ratio = l1_ratio)
  op <- par(no.readonly = TRUE)
  on.exit({ par(op); layout(1) })
  layout(matrix(1:2, nrow = 1), widths = c(1, 2))
  plot_coefficients(fit$w, sprintf('Weights, %d of %d non-zero', sum(fit$w != 0), p))
  plot_fit(fit$b + Z_star %*% fit$w, t_n,
           bquote(paste('Elastic Net fit, ', alpha, ' = ', .(alpha), ' and ', rho, ' = ', .(l1_ratio))))
  invisible(fit$w)
}

# alpha = 1e-6 and rho = 0.5: very weak penalty, close to OLS. The fit still
# follows the noise, although the weights are much smaller than the OLS ones.
fit_and_plot(alpha = 1e-6, l1_ratio = 0.5)

# alpha = 0.05 and rho = 0.5: strong penalty, the model is too rigid and underfits.
fit_and_plot(alpha = 0.05, l1_ratio = 0.5)

# alpha encodes the "complexity" of the model. Let us now fix an intermediate
# alpha and change the mix rho: Ridge keeps all the features with small
# weights, the Lasso sets many of them exactly to zero, the Elastic Net lies in
# between. rho encodes the sparsity of the fit.
fit_and_plot(alpha = 0.001, l1_ratio = 0)    # Ridge
fit_and_plot(alpha = 0.001, l1_ratio = 1)    # Lasso
fit_and_plot(alpha = 0.001, l1_ratio = 0.5)  # Elastic Net


# -----------------------------------------------------------------------------
# ElasticNet from Scikit-Learn
#
# In Python, ElasticNet solves the same objective inside a Pipeline (feature
# map + model). Here the "model object" is a list with the hyperparameters and,
# once fitted, the training mean of the features and the weights.
# -----------------------------------------------------------------------------
elastic_net <- function(alpha = 1, l1_ratio = 0.5) list(alpha = alpha, l1_ratio = l1_ratio)

enet_fit <- function(model, x, y, w_init = NULL, tol = 1e-8, max_iter = 100000) {
  # Fit the feature map + Elastic Net on (x, y).
  cen <- center(feature_map(x, p = p, l = l))
  fit <- elastic_net_cd(cen$Z, y, alpha = model$alpha, l1_ratio = model$l1_ratio,
                        w_init = w_init, tol = tol, max_iter = max_iter)
  # Intercept for the uncentered features, as scikit-learn reports it.
  model$coef <- fit$w
  model$intercept <- fit$b - sum(cen$mean * fit$w)
  model
}

enet_predict <- function(model, x) {
  as.vector(model$intercept + feature_map(x, p = p, l = l) %*% model$coef)
}

# Model Object.
model <- enet_fit(elastic_net(alpha = 0.001, l1_ratio = 0.5), x, t_n)
cat(sprintf('Elastic Net weights (alpha = 0.001, rho = 0.5): %d of %d non-zero, intercept = %.6f\n',
            sum(model$coef != 0), p, model$intercept))

# ElasticNetCV: choose alpha and rho by k-fold cross-validation. The training
# points are sorted by x, so the folds are shuffled: otherwise each fold would
# be a contiguous interval and the model would be validated on extrapolation.
kfold <- function(n, k = 5) split(sample(n), rep_len(seq_len(k), n))

elastic_net_cv <- function(x, y, l1_ratios, alphas, folds) {
  # Mean CV MSE for every (rho, alpha); alphas from the largest to the smallest
  # so each fit warm-starts from the previous one. max_iter is kept low to make
  # the loops fast in R: only the smallest alphas (whose CV error is large
  # anyway) stop before full convergence.
  alphas <- sort(alphas, decreasing = TRUE)
  mse_path <- array(0, dim = c(length(l1_ratios), length(alphas), length(folds)))
  for (i in seq_along(l1_ratios)) {
    for (k in seq_along(folds)) {
      test <- folds[[k]]
      w <- NULL
      for (a in seq_along(alphas)) {
        m <- enet_fit(elastic_net(alphas[a], l1_ratios[i]), x[-test], y[-test],
                      w_init = w, tol = 1e-6, max_iter = 500)
        w <- m$coef
        mse_path[i, a, k] <- mean((y[test] - enet_predict(m, x[test]))^2)
      }
    }
  }
  mse <- apply(mse_path, c(1, 2), mean)
  best <- arrayInd(which.min(mse), dim(mse))
  list(l1_ratios = l1_ratios, alphas = alphas, mse_path = mse_path, mse = mse,
       alpha = alphas[best[2]], l1_ratio = l1_ratios[best[1]])
}

# Coarser grid than in Python, to keep the R loops fast.
l1_ratios <- c(0.1, 0.5, 0.9, 1)
cv <- elastic_net_cv(x, t_n, l1_ratios, 10^seq(-6, 0, length.out = 30), kfold(n, 5))
# Refit on all the data with the selected hyperparameters.
model_cv <- enet_fit(elastic_net(cv$alpha, cv$l1_ratio), x, t_n)
cat(sprintf('Selected alpha: %.2e\n', cv$alpha))
cat(sprintf('Selected l1_ratio: %s\n', cv$l1_ratio))
cat(sprintf('Non-zero weights: %d of %d\n', sum(model_cv$coef != 0), p))

# Cross-validation error as a function of alpha, for each value of rho.
matplot(cv$alphas, t(cv$mse), type = 'l', lty = 1, lwd = 2, col = colors[seq_along(l1_ratios)],
        log = 'xy', xlab = expression(alpha), ylab = 'mean CV MSE', main = 'Cross-validation error')
abline(v = cv$alpha, lty = 2)
legend('topleft', legend = c(paste('rho =', l1_ratios), 'selected alpha'),
       col = c(colors[seq_along(l1_ratios)], 'black'), lty = c(rep(1, length(l1_ratios)), 2), lwd = 2)

# Predictions.
# Make the prediction on test set.
y_pred <- enet_predict(model_cv, xx)
cat(sprintf('Prediction length: %d\n', length(y_pred)))

plot(xx, yy, type = 'l', col = colors[1], lwd = 2, xlab = 'x', ylab = '',
     ylim = range(yy, y_pred, t_n),
     main = bquote(paste('Prediction ElasticNetCV, ', alpha, ' = ', .(signif(cv$alpha, 3)),
                         ' and ', rho, ' = ', .(cv$l1_ratio))))
lines(xx, y_pred, col = colors[2], lwd = 2)
points(x, t_n, pch = 19, col = colors[5])
legend('topright', legend = c('f(x)', 'pred'), col = colors[1:2], lwd = 2)

# Prediction Interval.
# The Elastic Net is not a Bayesian model, so there are no credible intervals.
# We use the bootstrap instead: refit the model (with the selected alpha and
# rho) on resamples of the training set and look at the spread of the
# predictions. This is a frequentist variability band, which ignores the bias
# introduced by the penalty.
n_boot <- 100
y_hat_samples <- matrix(0, nrow = length(xx), ncol = n_boot)
for (i in seq_len(n_boot)) {
  idx <- sample(n, n, replace = TRUE)
  m <- enet_fit(elastic_net(cv$alpha, cv$l1_ratio), x[idx], t_n[idx], tol = 1e-6, max_iter = 1000)
  y_hat_samples[, i] <- enet_predict(m, xx)
}
# Compute the mean of the sample.
y_hat <- rowMeans(y_hat_samples)
# Compute the standard deviation of the sample (population sd, as numpy's std).
y_hat_sd <- sqrt(rowMeans((y_hat_samples - y_hat)^2))

# Corridor associated with two standard deviations. With only 20 training
# points the band is wide near the edges and in the gaps between the points.
lower <- y_hat - 2*y_hat_sd
upper <- y_hat + 2*y_hat_sd
plot(x, t_n, pch = 19, col = colors[1], xlim = range(xx), ylim = c(-3, 3),
     xlab = 'x', ylab = '', main = 'Prediction & Bootstrap Interval')
polygon(c(xx, rev(xx)), c(lower, rev(upper)), col = adjustcolor('green', 0.3), border = NA)
lines(xx, yy, col = 'red', lwd = 2)
lines(xx, y_pred, col = 'darkgreen', lwd = 2)
legend('topright', legend = c('training data', 'Bootstrap Interval', 'f(x)', 'pred'),
       col = c(colors[1], adjustcolor('green', 0.3), 'red', 'darkgreen'),
       pch = c(19, 15, NA, NA), lty = c(NA, NA, 1, 1), pt.cex = c(1, 2, 1, 1), lwd = 2)


# -----------------------------------------------------------------------------
# Summary & Final Example
#
# A self-contained example in the setting the Elastic Net was designed for:
# more features than samples (p > n), with groups of correlated features of
# which only a few are relevant (e.g. a panel of biomarkers measured on a small
# number of subjects). Ridge, Lasso and Elastic Net are all fitted with the
# coordinate descent above, choosing alpha (and rho) by 5-fold CV.
# -----------------------------------------------------------------------------

# Construct Training Set.
# Number of training points.
n <- 100
# Number of features.
p <- 200
# Groups of correlated features: 5 groups of 10 features each.
n_groups <- 5
group_size <- 10
# Correlation of the features within a group.
rho_group <- 0.95
# Error standard deviation.
sigma_n <- 5

generate_features <- function(n_samples) {
  # Correlated groups (shared latent factor + noise) followed by independent features.
  latent <- matrix(rnorm(n_samples * n_groups), nrow = n_samples)
  noise <- matrix(rnorm(n_samples * n_groups * group_size), nrow = n_samples)
  grouped <- sqrt(rho_group) * latent[, rep(seq_len(n_groups), each = group_size)] + sqrt(1 - rho_group) * noise
  independent <- matrix(rnorm(n_samples * (p - n_groups * group_size)), nrow = n_samples)
  cbind(grouped, independent)
}

# True weights: only the first 3 groups are relevant, every feature of a group has the same weight.
w_true <- numeric(p)
w_true[1:10] <- 1
w_true[11:20] <- -1
w_true[21:30] <- 0.5

X <- generate_features(n)
# Observed target variable.
y <- as.vector(X %*% w_true + rnorm(n, sd = sigma_n))

# Construct Test Set.
n_star <- 1000
X_star <- generate_features(n_star)
y_star <- as.vector(X_star %*% w_true + rnorm(n_star, sd = sigma_n))

# Define Models.
# Standardize with the training mean and (population) standard deviation, as StandardScaler.
X_mean <- colMeans(X)
X_sd <- sqrt(colMeans(sweep(X, 2, X_mean)^2))
X_s <- scale(X, center = X_mean, scale = X_sd)
X_star_s <- scale(X_star, center = X_mean, scale = X_sd)

linear_cv <- function(X, y, l1_ratios, n_alphas = 20, k = 5) {
  # Choose (alpha, rho) by k-fold CV and refit on all the data. rho = 0 uses
  # the Ridge closed form on a fixed grid; otherwise coordinate descent along a
  # path from alpha_max (all weights zero) down to alpha_max / 1000, as scikit-learn.
  fit_one <- function(Xtr, ytr, alpha, rho, w_init) {
    if (rho == 0) ridge_closed_form(Xtr, ytr, alpha)
    else elastic_net_cd(Xtr, ytr, alpha, rho, w_init = w_init, tol = 1e-4, max_iter = 1000)
  }
  folds <- kfold(nrow(X), k)
  best <- list(mse = Inf)
  for (rho in l1_ratios) {
    alphas <- if (rho == 0) {
      10^seq(2, -4, length.out = n_alphas)
    } else {
      max(abs(crossprod(X, y - mean(y)))) / (nrow(X) * rho) * 10^seq(0, -3, length.out = n_alphas)
    }
    mse <- matrix(0, nrow = n_alphas, ncol = k)
    for (j in seq_len(k)) {
      test <- folds[[j]]
      Xtr <- scale(X[-test, ], scale = FALSE)
      w <- NULL
      for (a in seq_len(n_alphas)) {
        fit <- fit_one(Xtr, y[-test], alphas[a], rho, w)
        w <- fit$w
        pred <- fit$b + scale(X[test, , drop = FALSE], center = attr(Xtr, 'scaled:center'), scale = FALSE) %*% w
        mse[a, j] <- mean((y[test] - pred)^2)
      }
    }
    a_best <- which.min(rowMeans(mse))
    if (min(rowMeans(mse)) < best$mse) best <- list(mse = min(rowMeans(mse)), alpha = alphas[a_best], l1_ratio = rho)
  }
  Xc <- scale(X, scale = FALSE)
  fit <- fit_one(Xc, y, best$alpha, best$l1_ratio, NULL)
  list(coef = fit$w, intercept = fit$b - sum(attr(Xc, 'scaled:center') * fit$w),
       alpha = best$alpha, l1_ratio = best$l1_ratio)
}

# Model Fit + Predictions.
models <- list(
  'Ridge' = linear_cv(X_s, y, l1_ratios = 0, n_alphas = 60),
  'Lasso' = linear_cv(X_s, y, l1_ratios = 1),
  'Elastic Net' = linear_cv(X_s, y, l1_ratios = c(0.1, 0.5, 0.9, 0.95))
)
y_preds <- list()
for (name in names(models)) {
  m <- models[[name]]
  # Make the prediction on test set.
  y_preds[[name]] <- as.vector(m$intercept + X_star_s %*% m$coef)
  selected <- which(m$coef != 0)
  cat(sprintf('%-12s test MSE = %5.2f   non-zero weights = %3d   relevant features selected = %2d of %d\n',
              name, mean((y_star - y_preds[[name]])^2), length(selected),
              sum(w_true[selected] != 0), sum(w_true != 0)))
}
cat(sprintf('Elastic Net: alpha = %.3f, l1_ratio = %s\n', models[['Elastic Net']]$alpha, models[['Elastic Net']]$l1_ratio))

# Visualize Results.
# Estimated vs true weights of the first 60 features (the 5 correlated groups,
# separated by dotted lines, and 10 independent features). The Lasso tends to
# select only some features of each relevant group; the Elastic Net keeps the
# groups together (grouping effect); Ridge keeps all the features.
op <- par(mfrow = c(3, 1), mar = c(3, 4, 3, 1))
for (name in names(models)) {
  # Weights on the original scale of the features.
  coef <- models[[name]]$coef / X_sd
  ylim <- range(-2, 2, coef[1:60])
  barplot(rbind(w_true[1:60], coef[1:60]), beside = TRUE, col = c(alpha_col(colors[5], 0.5), colors[1]),
          border = NA, space = c(0, 0.5), ylim = ylim, ylab = 'weight', main = paste(name, 'weights'))
  abline(v = (1:n_groups) * group_size * 2.5 + 0.25, lty = 3)
  abline(h = 0)
  legend('topright', legend = c('true', name), fill = c(alpha_col(colors[5], 0.5), colors[1]), border = NA)
}
par(op)

# Predictions on the test set.
op <- par(mfrow = c(1, 3))
lims <- range(y_star)
for (name in names(models)) {
  plot(y_star, y_preds[[name]], pch = 19, col = alpha_col(colors[1], 0.4), xlim = lims, ylim = lims,
       xlab = 'observed', ylab = 'predicted',
       main = sprintf('%s, test MSE = %.2f', name, mean((y_star - y_preds[[name]])^2)))
  abline(0, 1, col = colors[2], lwd = 2)
}
par(op)

if (!interactive()) invisible(dev.off())
