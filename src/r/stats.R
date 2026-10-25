# Hypothesis testing and bio-statistical evaluations

#' Two-group comparison (Welch t-test or Wilcoxon)
compare_groups <- function(x, y, parametric = TRUE) {
  if (parametric) stats::t.test(x, y) else stats::wilcox.test(x, y)
}

#' Cohen's d effect size
cohens_d <- function(x, y) {
  nx <- length(x)
  ny <- length(y)
  pooled_sd <- sqrt(((nx - 1) * stats::var(x) + (ny - 1) * stats::var(y)) / (nx + ny - 2))
  (mean(x) - mean(y)) / pooled_sd
}
