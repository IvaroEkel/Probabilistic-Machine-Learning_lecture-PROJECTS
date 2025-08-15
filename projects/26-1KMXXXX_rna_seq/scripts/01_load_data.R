# R/01_load_data.R
# Load libraries, raw counts, and sample annotations

# ------------------ Libraries ------------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(MASS)
  library(ggplot2)
  library(ggrepel)
  library(pheatmap)
  library(RColorBrewer)
  library(pbapply)
  library(matrixStats)
})

# ------------------ Set working directory ------------------
setwd(here::here())  # safer than hardcoding

# ------------------ Load raw counts ------------------
counts_raw <- read.delim(
  "data/GSE126848_raw_counts_GRCh38.p13_NCBI.tsv",
  header = TRUE, row.names = 1, check.names = FALSE
)
log_info("Genes (rows):", nrow(counts_raw))
log_info("Samples (cols):", ncol(counts_raw))

# ------------------ Sample IDs ------------------

# this information stems from the metadata file (also given in the repository)

nash_samples <- c(
  "GSM3615308","GSM3615309","GSM3615310","GSM3615311","GSM3615312",
  "GSM3615313","GSM3615314","GSM3615315","GSM3615316","GSM3615317",
  "GSM3615318","GSM3615319","GSM3615320","GSM3615321","GSM3615322"
)
healthy_samples <- c(
  "GSM3615323","GSM3615324","GSM3615325","GSM3615326","GSM3615327",
  "GSM3615328","GSM3615329","GSM3615330","GSM3615331","GSM3615332",
  "GSM3615333","GSM3615334","GSM3615335","GSM3615336"
)
selected_samples <- c(nash_samples, healthy_samples)

# Check presence in count matrix
missing_in_counts <- setdiff(selected_samples, colnames(counts_raw))
if (length(missing_in_counts) > 0) {
  warning("Samples missing in count matrix and will be dropped: ",
          paste(missing_in_counts, collapse = ", "))
}
present_samples <- intersect(selected_samples, colnames(counts_raw))
stopifnot(length(present_samples) > 0)

# ------------------ Create colData ------------------
counts <- counts_raw[, present_samples, drop = FALSE]
colData <- tibble(
  sample_id = present_samples,
  condition = ifelse(present_samples %in% nash_samples, "NASH", "Healthy")
) %>%
  arrange(condition, sample_id) %>%
  group_by(condition) %>%
  mutate(sample = paste0(condition, "_", row_number())) %>%
  ungroup() %>%
  select(sample, condition, sample_id)

counts <- counts[, colData$sample_id, drop = FALSE]
colnames(counts) <- colData$sample
rownames(colData) <- colData$sample
colData$condition <- factor(colData$condition, levels = c("Healthy", "NASH"))

log_info("Samples by condition:")
print(colData %>% count(condition))