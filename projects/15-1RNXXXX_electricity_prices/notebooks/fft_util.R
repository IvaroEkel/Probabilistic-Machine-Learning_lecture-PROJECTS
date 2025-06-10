library(dplyr)
library(rlang)

# 1) compute_fft(): given a data frame, a time‐column, and a value‐column,
#                  returns a tibble with (freq, amplitude).
compute_fft <- function(data, time_col, value_col) {
  # Capture the “time” and “value” columns via enquo()
  time_col  <- enquo(time_col)
  value_col <- enquo(value_col)
  
  # Pull out just those two columns, and rename them internally to "time" / "value"
  df <- data %>%
    select(
      time  = !!time_col,
      value = !!value_col
    ) %>%
    arrange(time)   # (ensure sorted in time order)
  
  # Extract as vectors
  t_vec <- df$time
  x_vec <- df$value
  n     <- length(x_vec)
  
  # Compute sampling interval (assumes uniform spacing)
  dt <-as.numeric(t_vec[2] - t_vec[1])    # assume uniform spacing
  fs <- 1 / dt                 # sampling frequency in Hz
  
  # Raw (complex) FFT
  X  <- fft(x_vec)
  
  # Keep only the first (n/2 + 1) bins (one‐sided spectrum)
  n_half <- floor(n/2) + 1
  
  # Build the frequency axis (0, fs/n, 2·fs/n, …, (n_half-1)·fs/n)
  freq <- (0:(n_half - 1)) * (fs / n)
  
  # Compute one‐sided amplitude: (2/n)·|X[k]| for k = 1 … n_half
  # (Note: for k=0 and k=n/2 it’s technically not “double”, 
  #  but the formula below is the usual convention.)
  amplitude <- (2 / n) * Mod(X)[1:n_half]
  
  # Return as a tibble
  tibble(
    freq      = freq,
    amplitude = amplitude
  )
}



