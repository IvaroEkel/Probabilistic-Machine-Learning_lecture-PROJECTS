# R/04_qc_plots.R
# Visual QC: PCA, Boxplot, Correlation heatmap, Variance~Mean

qc_save_plots <- function(logcounts, colData, outdir_figs) {
  ensure_dir(outdir_figs)

  # Align design to expression matrix (avoid join/name issues)
  sn <- trimws(colnames(logcounts))
  cd <- as.data.frame(colData)
  if (!"sample" %in% names(cd)) cd$sample <- rownames(cd)
  cd$sample <- trimws(as.character(cd$sample))
  design <- cd[, c("sample","condition"), drop = FALSE]

  # Set check with clear diagnostics
  if (!setequal(sn, design$sample)) {
    missing_in_cd <- setdiff(sn, design$sample)
    missing_in_sn <- setdiff(design$sample, sn)
    if (length(missing_in_cd)) message("Only in logcounts: ", paste(missing_in_cd, collapse = ", "))
    if (length(missing_in_sn)) message("Only in colData: ", paste(missing_in_sn, collapse = ", "))
    stop("Sample name mismatch between logcounts and colData.")
  }

  # Reorder design to exactly match logcounts order
  design <- design[match(sn, design$sample), , drop = FALSE]
  stopifnot(identical(design$sample, sn))

  # ---- PCA ----
  pca <- prcomp(t(logcounts), center = TRUE, scale. = FALSE)
  percentVar <- round(100 * (pca$sdev^2 / sum(pca$sdev^2)), 2)
  pca_df <- as.data.frame(pca$x)
  pca_df$sample <- sn
  pca_df$condition <- factor(design$condition, levels = c("Healthy", "NASH"))

  g_pca <- ggplot2::ggplot(pca_df, ggplot2::aes(PC1, PC2, color = condition)) +
    ggplot2::geom_point(size = 3) +
    ggplot2::labs(
      title = "PCA: log2(normalized counts + 1)",
      x = paste0("PC1: ", percentVar[1], "%"),
      y = paste0("PC2: ", percentVar[2], "%")
    ) +
    ggplot2::theme_minimal()
  ggplot2::ggsave(file.path(outdir_figs, "pca.png"), g_pca, width = 6, height = 4, dpi = 150)

  # ---- Boxplot ----
  g_box <- ggplot2::ggplot(stack(as.data.frame(logcounts)),
                           ggplot2::aes(x = ind, y = values)) +
    ggplot2::geom_boxplot(outlier.size = 0.4) +
    ggplot2::theme_minimal() +
    ggplot2::labs(x = "", y = "log2(norm+1)", title = "Log2 Normalized Counts")
  ggplot2::ggsave(file.path(outdir_figs, "boxplot_logcounts.png"), g_box, width = 8, height = 4, dpi = 150)

  # ---- Correlation heatmap (ordered by condition) ----
  sample_cor <- stats::cor(logcounts, method = "pearson")
  ord <- design %>% dplyr::arrange(condition) %>% dplyr::pull(sample)
  sample_cor_ord <- sample_cor[ord, ord]
  pheatmap::pheatmap(
    sample_cor_ord,
    main = "Sample Correlation (ordered by condition)",
    display_numbers = TRUE,
    cluster_rows = FALSE, cluster_cols = FALSE,
    color = grDevices::colorRampPalette(rev(RColorBrewer::brewer.pal(9, "Blues")))(100),
    filename = file.path(outdir_figs, "cor_heatmap.png")
  )
}

# ------- Variance~Mean -------- 
disp_save_plots <- function(norm_counts, outdir_figs) {
  ensure_dir(outdir_figs)

  mean_nc <- rowMeans(norm_counts)
  var_nc  <- matrixStats::rowVars(as.matrix(norm_counts))
  var_df  <- data.frame(mean = mean_nc, variance = var_nc) %>% dplyr::filter(mean > 0)

  g_vm <- ggplot2::ggplot(var_df, ggplot2::aes(x = mean, y = variance)) +
    ggplot2::geom_point(alpha = 0.4, size = 0.8) +
    ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    ggplot2::scale_x_log10() + ggplot2::scale_y_log10() +
    ggplot2::labs(
      title = "Empirical Variance vs Mean (Normalized Counts)",
      subtitle = "Red dashed: Poisson (Var = Mean)",
      x = "Mean (log10)", y = "Variance (log10)"
    ) + ggplot2::theme_minimal()

  ggplot2::ggsave(file.path(outdir_figs, "var_vs_mean.png"), g_vm, width = 6, height = 4, dpi = 150)
}