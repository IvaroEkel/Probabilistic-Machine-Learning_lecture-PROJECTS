library(tidyverse)

load_electricity_prices <- function() {
  ep_2015_2020 <- read_delim("/home/niclas/uni_leipzig/07_Veranstaltungen_SoSe25/ProbML/15-1RNXXXX_electricity_prices/data/electricity-prices-2015-2020.csv", 
                             delim = ";", escape_double = FALSE,
                             locale = locale(decimal_mark = ",", grouping_mark = "."), 
                             trim_ws = TRUE,
                             show_col_types = FALSE)
  ep_2020_2025 <- read_delim("/home/niclas/uni_leipzig/07_Veranstaltungen_SoSe25/ProbML/15-1RNXXXX_electricity_prices/data/electricity-prices-2020-2025.csv", 
                             delim = ";", escape_double = FALSE,
                             locale = locale(decimal_mark = ",", grouping_mark = "."), 
                             trim_ws = TRUE,
                             show_col_types = FALSE)
  
  ep <- rbind(ep_2015_2020, ep_2020_2025)
  return (ep)
}

prepare_data <- function(ep) {
  start_date <- ymd("2018-10-01")
  # rename the columns
  names_ep <- c("Date_from", "Date_to", "DELU", "avg_ngb", "BE", "DK1", "DK2", "FR", "NL", "NO2", "AT", "PL", "SW4", "CH", "IT", "SI", "HU" )
  names(ep) <- names_ep
  
  # remove the not named columns
  ep <- ep[, 1:length(names_ep)]
  
  # adjust the dates 
  ep_prepared <- ep %>%
    mutate(Date_from = dmy_hm(Date_from),
           Date_to = dmy_hm(Date_to)) %>%
    mutate(Date_from = as_datetime(Date_from),
           Date_to = as_datetime(Date_to)) %>%
    filter(Date_from >= start_date)
  
  # convert characters to double, adjust for comma as decimal separator
  ep_prepared <- ep_prepared %>%
    mutate(DELU =  as.numeric(sub(",", ".", DELU, fixed = TRUE)),
           avg_ngb =  as.numeric(sub(",", ".", avg_ngb, fixed = TRUE)),
           AT =  as.numeric(sub(",", ".", AT, fixed = TRUE)),
           PL = as.numeric(sub(",", ".", PL, fixed = TRUE)),
           SW4 = as.numeric(sub(",", ".", SW4, fixed = TRUE)),
           SI = as.numeric(sub(",", ".", SI, fixed = TRUE)),
           HU = as.numeric(sub(",", ".", HU, fixed = TRUE)))
  
  # delete duplicate values which come from switching 
  # from winter and summer time
  ep_prepared <- ep_prepared %>%
    distinct(Date_from, .keep_all = TRUE)
  
  return(ep_prepared)
}