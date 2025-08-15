# R/00_utils.R
# Small helper utilities used across the pipeline

# Pretty timestamped logging to console
log_info <- function(...) {
  cat(sprintf("[%s] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")), ..., "\n")
}

# Assert that a matrix/data.frame is non-empty
assert_nonempty_matrix <- function(m, name = deparse(substitute(m))) {
  if (is.null(dim(m)) || any(dim(m) == 0)) {
    stop(sprintf("Matrix '%s' is empty (dim = %s).", name,
                 paste0(dim(m), collapse = "×")))
  }
}

# Ensure a directory exists (no error if it already does)
ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}