here::i_am("main.R")

# main.R — Orchestrates the modular NB-GLM RNA-seq pipeline

suppressPackageStartupMessages({
  library(tidyverse); library(here); library(conflicted)
  # prefer dplyr verbs if other packages mask them
  conflict_prefer("select","dplyr", quiet = TRUE)
  conflict_prefer("filter","dplyr", quiet = TRUE)
  conflict_prefer("rename","dplyr", quiet = TRUE)
  conflict_prefer("arrange","dplyr", quiet = TRUE)
  conflict_prefer("mutate","dplyr", quiet = TRUE)
  conflict_prefer("lag","dplyr", quiet = TRUE)
  conflict_prefer("count","dplyr", quiet = TRUE)
})

# ---------- Config ----------
cfg <- list(
  counts_path  = here::here("data/GSE126848_raw_counts_GRCh38.p13_NCBI.tsv"),
  out_tables   = here::here("outputs/tables"),
  out_figs     = here::here("outputs/figures"),
  mean_cutoff  = 5,
  mad_quantile = 0.99,
  padj_cutoff  = 0.05,
  lfc_cutoff   = 1
)

# ---------- Source modules ----------
source(here::here("scripts/00_utils.R"))
source(here::here("scripts/01_load_data.R"))  # defines counts_raw, counts, colData
source(here::here("scripts/02_filter_qc.R"))
source(here::here("scripts/03_normalize.R"))
source(here::here("scripts/04_qc_plots.R"))
source(here::here("scripts/05_nb_glm.R"))
source(here::here("scripts/06_export.R"))

# ---------- Ensure output folders ----------
ensure_dir(cfg$out_tables)
ensure_dir(cfg$out_figs)

# ---------- Pipeline ----------
log_info("Filter counts (QC)...")
filt <- filter_counts(counts, mean_cutoff = cfg$mean_cutoff, mad_q = cfg$mad_quantile)

log_info("Normalize (median-of-ratios)...")
norm <- normalize_mor(filt$counts)  # size_factors, norm_counts, logcounts

log_info("QC plots (PCA, Boxplot, Heatmap)...")
qc_save_plots(norm$logcounts, colData, cfg$out_figs)

log_info("Dispersion plot (Var~Mean)...")
disp_save_plots(norm$norm_counts, cfg$out_figs)

log_info("Fit NB-GLM with offset...")
res <- nb_glm_with_offset(filt$counts, colData, norm$size_factors)

# safety: ensure 'gene' column present
if (!"gene" %in% names(res) && !is.null(rownames(res))) {
  res <- dplyr::mutate(res, gene = rownames(res))
}

log_info("Export results & volcano plot...")
export_deg_results(
  results_df   = res,
  norm_counts  = norm$norm_counts,
  log2fc_cutoff = cfg$lfc_cutoff,
  padj_cutoff   = cfg$padj_cutoff,
  table_path    = file.path(cfg$out_tables, "nb_glm_differential_expression_results.csv"),
  volcano_path  = file.path(cfg$out_figs, "volcano.png")
)

log_info("Done.")