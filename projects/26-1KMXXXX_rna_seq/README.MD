
# Differential Gene Expression detection via Generalized Linear Models

This repository contains a probabilistic machine learning project for detecting differentially expressed genes using generalized linear models. 

# Dataset 

The dataset is publicy available under the Gene Expression Omnibus (GEO) Acession number: [GSE126848](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126848)

The original Pubilcation can be found [here](https://pubmed.ncbi.nlm.nih.gov/30653341/)

# Repository Tree Structure

```
project/
│ 
├─ data/  
│  ├─ GSE126848_raw_counts_GRCh38.p13_NCBI.tsv   # count matrix
│  └─ sample_lists.yaml                          # sample mapping 
│  └─ metadata.csv
│ 
├─ outputs/  
│  ├─ figures/  
│  └─ tables/  
│ 
├─ scripts/  
│  ├─ 00_utils.R  
│  ├─ 01_load_data.R  
│  ├─ 02_filter_qc.R  
│  ├─ 03_normalize.R  
│  ├─ 04_qc_plots.R  
│  ├─ 05_nb_glm.R  
│  └─ 06_export.R  
│ 
└─ main.R
└─ requirements.txt
└─ README.md
│ 
└─ report.pdf
```

# Using this Repository

## Dependencies 

Please refer to requirements.txt

## Setup and run piepline

1. Clone the Repository: 
```git clone https://github.com/MaxKolczynski/Probabilistic-Machine-Learning_lecture-PROJECTS.git```

2. go the the root directory: ```cd projects/26-1KMXXXX_rna_seq```  

3. Simply execute ```Rscript main.R``` to get the entire piepline running. It will automatically fetch the modules from the ``scripts/``folder. Figures and tables that can be found in the ```outputs``` folder. 

## Notes on the code modules

### 00_utils.R  

Small shared helpers
-	log_info(...): timestamped console logging.
-	assert_nonempty_matrix(m, name): stops if a matrix/data.frame is empty (clear error message with dims).
-	ensure_dir(path): creates directories recursively if missing.

Used throughout for robust checks and clean console output.

### 01_load_data.R  

- Loads packages: ```tidyverse```, ```MASS```, ```ggplot2```, ```ggrepel```, ```pheatmap```, ```RColorBrewer```, ```pbapply```, ```matrixStats```
-	Reads raw count matrix and sample metadata (assumes columns are sample IDs)
-	Aligns counts to the present samples; builds colData with sample, condition (factor with levels Healthy, NASH)
-	Renames columns to stable sample labels (e.g., Healthy_1, NASH_1) and sets matching rownames in colData
-	Prints a brief condition summary to the console

Output objects: counts (genes × samples), colData (rows = samples, columns: sample, condition).

### 02_filter_qc.R  

Gene-level QC and filtering to remove noise and obvious artifacts.
	-	Quick diagnostics: library-size summary, number of all-zero genes.
	-	Step 1: keep genes with nonzero total counts.
	-	Step 2: remove outlier genes via a MAD filter on per-gene statistics (upper quantile mad_q, default 0.99).
	-	Step 3: require minimum mean count per gene (mean_cutoff, default 5).
	-	Rounds to integer and returns a dense matrix.
	-	Prints a clear filtering summary (genes before/after each step).

Output object: counts_filt.

### 03_normalize.R  

Manual median-of-ratios (DESeq-style) size-factor normalization + log2 transform.
-	Adds pseudocount (+1) to avoid log(0).
-	Computes geometric mean per gene, forms per-sample ratios (counts / geo mean).
-	Size factor for each sample = median of its ratios.
-	Normalized counts = raw counts / size factor.
 - logcounts = log2(norm_counts + 1) for visualization.
-	Prints size-factor vector and summary.

Output objects: size_factors (named vector), norm_counts, logcounts.

### 04_qc_plots.R  

Generates and saves core QC visualizations.
- Robustly aligns expression matrices and colData (diagnoses missing/mismatched sample names).
- PCA on logcounts (center/scale): plots PC1–PC2 with samples colored by condition; saves pca.png.
- Boxplot of logcounts distributions per sample; saves boxplot_logcounts.png.
- Correlation heatmap (sample–sample Pearson cor(logcounts)), with a qualitative palette; saves cor_heatmap.png.
- Mean–variance plot on norm_counts (log–log scale) with Poisson reference line (Var=Mean); saves var_vs_mean.png.

Side effects: creates outdir_figs if needed and writes figure files there.


### 05_nb_glm.R  

Per-gene Negative Binomial GLM with library-size offset (median-of-ratios).
-	Strict alignment of counts_filt, colData, and size_factors; stops early with a detailed message if inconsistent.
-	Builds design: ~ condition with Healthy as reference (factor levels set earlier).
-	For each gene: fits MASS::glm.nb(counts ~ condition + offset(log(size_factor))).
-	Extracts the conditionNASH coefficient as log2 fold change proxy (logFC), its SE, and Wald p-value.
-	Adjusts p-values via Benjamini–Hochberg (padj).
-	Prints DEG counts (FDR<0.05; up/down).

Output object: data.frame with columns: gene, logFC, se, pval, padj 

### 06_export.R  

Exports results and (optionally) creates a publication-style volcano plot.
-	Computes baseMean per gene from norm_counts.
-	Cleans and augments results: keeps finite (logFC, pval, padj), adds baseMean and a simple Wald stat = logFC / se.
-	Writes a CSV table to outputs/tables/nb_glm_differential_expression_results.csv (path configurable).
-	Volcano plot: x = logFC, y = -log10(padj).
-	Colors points by regulation (Up/Down/NS) using thresholds |log2FC| ≥ log2fc_cutoff and padj ≤ padj_cutoff (defaults: 1 and 0.05).
-	Adds dashed guides at the chosen cutoffs; x-limits clipped to a reasonable range.
-	Saves to outputs/figures/volcano_plot.png (path configurable).

Return value: returns the exported DEG table  ￼




