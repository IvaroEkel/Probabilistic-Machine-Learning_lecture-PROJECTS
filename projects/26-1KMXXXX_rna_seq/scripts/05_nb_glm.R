# R/05_nb_glm.R
# Gene-wise Negative Binomial GLM with library-size offset (median-of-ratios)

nb_glm_with_offset <- function(counts_filt, colData, size_factors) {
  # counts_filt: integer matrix (genes x samples) after QC filters
  # colData: data.frame with rows named by sample, columns: sample, condition
  # size_factors: named numeric vector, names == colnames(counts_filt)

  # ---- Robust alignment between counts_filt, colData, and size_factors ----
  sn <- trimws(colnames(counts_filt))
  cd <- as.data.frame(colData)
  if (!"sample" %in% names(cd)) cd$sample <- rownames(cd)
  cd$sample <- trimws(as.character(cd$sample))

  # Check set equality and diagnose differences
  if (!setequal(sn, cd$sample)) {
    missing_in_cd <- setdiff(sn, cd$sample)
    missing_in_sn <- setdiff(cd$sample, sn)
    if (length(missing_in_cd)) message("Only in counts_filt: ", paste(missing_in_cd, collapse = ", "))
    if (length(missing_in_sn)) message("Only in colData: ", paste(missing_in_sn, collapse = ", "))
    stop("Sample name mismatch between counts_filt and colData.")
  }

  # Reorder design table to match counts column order exactly
  design_tbl <- cd[, c("sample","condition"), drop = FALSE]
  design_tbl <- design_tbl[match(sn, design_tbl$sample), , drop = FALSE]
  stopifnot(identical(design_tbl$sample, sn))

  # Ensure size_factors are in the same order and complete
  sf <- size_factors
  if (!setequal(names(sf), sn)) {
    missing_sf <- setdiff(sn, names(sf))
    extra_sf   <- setdiff(names(sf), sn)
    if (length(missing_sf)) message("Missing size_factors for: ", paste(missing_sf, collapse = ", "))
    if (length(extra_sf))   message("Unused size_factors for: ", paste(extra_sf, collapse = ", "))
    stop("Size factor names do not match counts columns.")
  }
  sf <- sf[sn]

  pboptions(type = "txt", style = 3)

  gene_list <- rownames(counts_filt)

  glm_results <- pblapply(gene_list, function(g) {
    y <- counts_filt[g, ]
    df <- tibble::tibble(
      counts = as.numeric(y),
      sample = names(y)
    ) %>%
      dplyr::left_join(design_tbl, by = "sample") %>%
      dplyr::mutate(
        condition   = stats::relevel(factor(condition), ref = "Healthy"),
        size_factor = sf[sample]
      )

    # Skip if metadata/offset missing
    if (any(is.na(df$condition)) || any(is.na(df$size_factor))) {
      return(tibble::tibble(gene = g, logFC = NA_real_, se = NA_real_, pval = NA_real_))
    }

    # Fit NB-GLM with offset
    fit <- try(MASS::glm.nb(counts ~ condition + offset(log(size_factor)), data = df, link = log),
               silent = TRUE)
    if (inherits(fit, "try-error")) {
      return(tibble::tibble(gene = g, logFC = NA_real_, se = NA_real_, pval = NA_real_))
    }

    coefs <- summary(fit)$coefficients
    if (!("conditionNASH" %in% rownames(coefs))) {
      return(tibble::tibble(gene = g, logFC = NA_real_, se = NA_real_, pval = NA_real_))
    }

    tibble::tibble(
      gene = g,
      logFC = coefs["conditionNASH","Estimate"],
      se    = coefs["conditionNASH","Std. Error"],
      pval  = coefs["conditionNASH","Pr(>|z|)"]
    )
  })

  res <- dplyr::bind_rows(glm_results) %>%
    dplyr::mutate(padj = p.adjust(pval, method = "BH"))

  # Console summary for the report
  message("DEGs (FDR < 0.05): ", sum(res$padj < 0.05, na.rm = TRUE),
          " | Up: ", sum(res$padj < 0.05 & res$logFC > 0, na.rm = TRUE),
          " | Down: ", sum(res$padj < 0.05 & res$logFC < 0, na.rm = TRUE))

  res
}