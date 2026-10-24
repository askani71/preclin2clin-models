source(file.path("..", "..", "src", "r", "stats.R"))

test_that("cohens_d is zero for identical groups", {
  x <- c(1, 2, 3, 4)
  expect_equal(cohens_d(x, x), 0)
})
