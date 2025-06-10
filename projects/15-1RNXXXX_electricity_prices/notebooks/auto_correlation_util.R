library(dplyr)
library(rlang)
library(purrr)
library(tibble)

# compute_autocorrelation(): 
#   - data  : a data.frame or tibble
#   - ...   : tidy‐select specification of one or more numeric columns
# Returns a tibble with columns (variable, lag, acf).
compute_autocorrelation <- function(data, ..., lag.max=NULL) {
  # Capture the columns to compute autocorrelation on
  cols_quos <- enquos(...)
  
  # Subset and rename internally to keep things simple
  df_sel <- data %>% 
    select(!!!cols_quos) %>% 
    # ensure we work with numeric vectors
    mutate(across(everything(), as.numeric))
  
  # For each selected column, compute its one‐sided autocorrelation via stats::acf()
  map_dfr(names(df_sel), function(var_name) {
    x   <- df_sel[[var_name]]
    # Compute autocorrelation; plot = FALSE so no graphics pop up
    acf_res <- acf(x, plot = FALSE, na.action = na.pass, lag.max=lag.max)
    
    # acf_res$lag and acf_res$acf are 3D arrays of shape (lag, 1, 1) for univariate series.
    lag_vec <- as.numeric(acf_res$lag[, 1, 1])
    acf_vec <- as.numeric(acf_res$acf[, 1, 1])
    
    tibble(
      variable = var_name,
      lag      = lag_vec,
      acf      = acf_vec
    )
  })
}

library(tibble)

# # 1) Create some toy data
# set.seed(42)
# n     <- 100
# time  <- 1:n
# y1    <- 0.8 * cumsum(rnorm(n))
# y2    <- sin(2*pi*time/10) + rnorm(n, sd = 0.3)
# df    <- tibble(time = time, series_1 = y1, series_2 = y2)
# 
# # 2) Compute autocorrelations of series_1 and series_2
# acf_tbl <- compute_autocorrelation(df, series_1, series_2)
# head(acf_tbl, 10)
# 
# # 3) You can now plot or otherwise analyze acf_tbl, for example:
# library(ggplot2)
# ggplot(acf_tbl %>% filter(variable == "series_1"), aes(x = lag, y = acf)) +
#   geom_bar(stat = "identity", width = 0.2) +
#   labs(title = "Autocorrelation of series_1", x = "Lag", y = "ACF") +
#   theme_minimal()