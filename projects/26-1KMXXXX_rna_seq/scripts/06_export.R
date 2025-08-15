# R/06_export.R
# Export results table + optional volcano plot

export_deg_results <- function(results_df, norm_counts, log2fc_cutoff = 1, padj_cutoff = 0.05,
                               table_path = "outputs/tables/nb_glm_differential_expression_results.csv",
                               volcano_path = "outputs/figures/volcano_plot.png") {
  stopifnot("gene" %in% names(results_df))
  
  base_means <- rowMeans(norm_counts)
  
  deg_table <- results_df %>%
    dplyr::filter(!is.na(logFC) & !is.na(pval) & !is.na(padj)) %>%
    dplyr::mutate(
      baseMean = base_means[gene],
      stat = logFC / se
    ) %>%
    dplyr::rename(
      log2FoldChange = logFC,
      lfcSE = se,
      pvalue = pval
    ) %>%
    dplyr::select(gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj) %>%
    dplyr::arrange(padj)
  
  dir.create(dirname(table_path), showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(deg_table, table_path)
  message("Results table written to: ", table_path)
  
  # Volcano plot
  res_df <- deg_table %>%
    dplyr::mutate(
      significance = dplyr::case_when(
        log2FoldChange >  log2fc_cutoff & padj < padj_cutoff ~ "up",
        log2FoldChange < -log2fc_cutoff & padj < padj_cutoff ~ "down",
        TRUE ~ "ns"
      )
    )
  
  top_genes <- res_df %>%
    dplyr::filter(padj < 1e-10 & abs(log2FoldChange) > 2) %>%
    dplyr::slice_max(order_by = -log10(padj) * abs(log2FoldChange), n = 30)
  
  p <- ggplot2::ggplot(res_df, ggplot2::aes(x = log2FoldChange, y = -log10(padj), color = significance)) +
    ggplot2::geom_point(alpha = 0.6, size = 1.2) +
    ggrepel::geom_text_repel(data = top_genes, ggplot2::aes(label = gene), size = 3, max.overlaps = 30) +
    ggplot2::scale_color_manual(values = c("up" = "red", "down" = "blue", "ns" = "grey")) +
    ggplot2::geom_vline(xintercept = c(-log2fc_cutoff, log2fc_cutoff), linetype = "dashed") +
    ggplot2::geom_hline(yintercept = -log10(padj_cutoff), linetype = "dashed") +
    ggplot2::coord_cartesian(xlim = c(-7, 7)) +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = "Volcano Plot: NASH vs Healthy",
                  x = "log2 Fold Change (NB-GLM coefficient)",
                  y = "-log10 adjusted p-value",
                  color = "Regulation")
  
  dir.create(dirname(volcano_path), showWarnings = FALSE, recursive = TRUE)
  ggplot2::ggsave(volcano_path, p, width = 8, height = 6)
  message("Volcano plot saved to: ", volcano_path)
  
  invisible(deg_table)
}