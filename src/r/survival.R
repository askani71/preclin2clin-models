# Survival analysis and time-to-event functions

#' Fit a Kaplan-Meier estimator
fit_km <- function(data, time_col = "time", event_col = "event", group_col = NULL) {
  rhs <- if (is.null(group_col)) "1" else group_col
  f <- stats::as.formula(sprintf("survival::Surv(%s, %s) ~ %s", time_col, event_col, rhs))
  survival::survfit(f, data = data)
}

#' Fit a Cox proportional hazards model
fit_cox <- function(data, covariates, time_col = "time", event_col = "event") {
  f <- stats::as.formula(sprintf(
    "survival::Surv(%s, %s) ~ %s", time_col, event_col, paste(covariates, collapse = " + ")
  ))
  survival::coxph(f, data = data)
}
