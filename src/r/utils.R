# Helper routines

#' Path to a file in data/processed
processed_path <- function(filename, root = ".") {
  file.path(root, "data", "processed", filename)
}

#' Read a Parquet dataset shared with the Python pipeline
read_processed <- function(filename, root = ".") {
  arrow::read_parquet(processed_path(filename, root))
}
