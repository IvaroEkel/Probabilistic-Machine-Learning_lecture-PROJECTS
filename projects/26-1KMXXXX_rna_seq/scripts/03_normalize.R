# R/03_normalize.R
# Manual median-of-ratios normalization (DESeq2-style) + log2 transform

normalize_mor <- function(counts_filt) {
  assert_nonempty_matrix(counts_filt, "counts_filt")

  # 1) Pseudocount to avoid log(0)
  counts_pseudo <- counts_filt + 1

  # 2) Geometric mean per gene
  geo_means <- apply(counts_pseudo, 1, function(x) exp(mean(log(x))))

  # 3) Ratios = counts / geo_mean (per gene)
  ratios <- sweep(counts_pseudo, 1, geo_means, "/")

  # 4) Size factor = median of ratios per sample
  size_factors <- apply(ratios, 2, median, na.rm = TRUE)
  names(size_factors) <- colnames(counts_filt)

  # 5) Normalized counts = counts / size_factor
  norm_counts <- sweep(counts_filt, 2, size_factors, "/")

  # 6) Log2 for visualization
  logcounts <- log2(norm_counts + 1)

  cat("=== Size factors (median-of-ratios) ===\n")
  print(round(size_factors, 3))
  cat("Summary:\n"); print(summary(size_factors)); cat("\n")

  list(size_factors = size_factors,
       norm_counts  = norm_counts,
       logcounts    = logcounts)
}