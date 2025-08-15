# R/02_filter_qc.R
# Gene-level filtering (QC): remove all-zero genes, high-MAD outliers, low-mean genes

filter_counts <- function(counts, mean_cutoff = 5, mad_q = 0.99) {
  assert_nonempty_matrix(counts, "counts")

  # Quick summaries for the report
  lib_sizes <- colSums(counts)
  cat("=== Library sizes (raw counts) ===\n"); print(summary(lib_sizes)); cat("\n")
  zero_genes <- sum(rowSums(counts) == 0)
  cat("Genes with zero counts across all samples:", zero_genes, "\n\n")

  # 1) keep genes with nonzero total counts
  keep_nonzero <- rowSums(counts) > 0
  counts_nz <- counts[keep_nonzero, , drop = FALSE]

  # 2) remove extreme-MAD outliers (top (1 - mad_q))
  mad_vals <- matrixStats::rowMads(as.matrix(counts_nz))
  keep_mad <- mad_vals < stats::quantile(mad_vals, mad_q, na.rm = TRUE)
  counts_mad <- counts_nz[keep_mad, , drop = FALSE]

  # 3) keep genes with sufficient mean counts
  keep_mean <- rowMeans(counts_mad) > mean_cutoff
  counts_filt <- round(as.matrix(counts_mad[keep_mean, , drop = FALSE]))

  # Report filtering impact
  cat("=== Filtering summary ===\n")
  cat("Genes before:", nrow(counts), "\n")
  cat(" - after nonzero:", nrow(counts_nz), "\n")
  cat(" - after MAD filter (", mad_q, "): ", nrow(counts_mad), "\n", sep = "")
  cat(" - after mean >", mean_cutoff, ": ", nrow(counts_filt), "\n\n", sep = "")

  list(counts = counts_filt)
}