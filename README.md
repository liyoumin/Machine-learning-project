# Consumer Perception Impacts on Olive Oil Choice: A Machine Learning Case Study

This repository contains the manuscript source, R scripts, and supporting figures for the project:

**Consumer Perception Impacts on Olive Oil Choice: A Case Study Using Machine Learning Approach**  
Author: **Youmin Li**

The project studies how consumer perceptions of olive oil attributes shape consumption choice, especially the choice between **extra virgin olive oil (EVOO)** and **refined olive oil (ROO)**. The analysis combines unsupervised learning, structural modeling, and supervised prediction to identify latent consumer segments and evaluate out-of-sample predictive performance.

- [Project page](https://liyoumin.github.io/personalweb/publication/preprint/)
- [Preprint PDF](https://liyoumin.github.io/personalweb/publication/preprint/preprint.pdf)
- [GitHub repository](https://github.com/liyoumin/Machine-learning-project)

---

## Research objective

The central research question is:

> How do consumer perceptions of quality, taste, price, health, and trust influence olive oil consumption choice?

The project uses survey and choice-experiment data from Spanish consumers to:

1. Reduce high-dimensional perception variables into interpretable latent factors.
2. Segment consumers into perception-based groups.
3. Estimate perception-consumption pathways using PLS-SEM.
4. Predict EVOO versus ROO choice using cross-validated machine learning models.
5. Compare interpretable linear models with more flexible models such as random forest, XGBoost, and GAM.

---

## Methodological workflow

The empirical pipeline has two main stages.

### Stage 1: Unsupervised learning and data structure

- Clean and standardize Likert-scale survey items.
- Select perception variables related to:
  - trust in EVOO and ROO,
  - taste perception,
  - price perception,
  - health perception,
  - consumption frequency and product choice.
- Use **exploratory factor analysis (EFA)** / PCA-style factor scoring to reduce dimensionality.
- Use KMO and Bartlett diagnostics to assess factorability.
- Apply **k-means clustering** to factor scores.
- Select the number of clusters using silhouette diagnostics.
- Label perception-based consumer segments, including groups such as:
  - price-sensitive consumers,
  - taste-driven consumers,
  - EVOO-favoring consumers,
  - ROO-favoring consumers,
  - health/trust-oriented consumers.

### Stage 2: Structural modeling and prediction

- Estimate a **partial least squares structural equation model (PLS-SEM)** to explain perception-consumption pathways.
- Compare predictive models under cross-validation:
  - logistic regression,
  - LASSO-logit,
  - random forest,
  - XGBoost,
  - GAM-logit.
- Evaluate classification performance using:
  - ROC AUC,
  - accuracy,
  - log-loss,
  - calibration diagnostics,
  - variable importance.

---

## Repository structure

```text
Machine-learning-project/
├── appendix/                         # Appendix figures and supporting outputs
├── figure/                           # Main manuscript figures
├── Machine_learning_proposal- Youmin Li.pdf
├── SEM-PLS.R                         # PLS-SEM and structural pathway analysis
├── cluster-prediction.R              # EFA/PCA, clustering, model training, CV, ROC, VIP
├── main.tex                          # LaTeX manuscript source
├── progress_models.RData             # Saved intermediate model objects
└── references.bib                    # Bibliography
```

---

## Main scripts

### `cluster-prediction.R`

This is the main machine learning pipeline. It performs:

- package loading and setup,
- data import,
- perception-variable selection,
- EFA diagnostics,
- factor extraction and factor-score export,
- k-means clustering,
- consumer-segment profiling,
- train/test and cross-validation setup,
- logistic, LASSO, random forest, XGBoost, and GAM model estimation,
- ROC curve generation,
- variable-importance plotting,
- output table and figure export.

### `SEM-PLS.R`

This script estimates the structural equation modeling component. It defines latent constructs for:

- taste perception,
- price perception,
- trust in EVOO,
- trust in ROO,
- negative health perception,
- EVOO consumption,
- ROO consumption.

It then estimates a PLS-SEM model, bootstraps path estimates, and supports subgroup or multi-group comparison by demographic variables such as gender, income, and region.

### `main.tex`

This file contains the LaTeX source for the paper manuscript.

---

## Data

The raw survey data are not currently included in this repository. The R script expects an Excel file named similar to:

```text
Base_v1.xlsx
```

In `cluster-prediction.R`, update the data path before running the analysis:

```r
DATA_PATH <- "data/Base_v1.xlsx"
dat0 <- readxl::read_xlsx(DATA_PATH) |> janitor::clean_names()
```

A recommended local structure is:

```text
Machine-learning-project/
├── data/
│   └── Base_v1.xlsx
├── outputs/
├── figure/
└── appendix/
```

Because the original script uses an absolute local path, users reproducing the analysis should replace it with a relative project path such as `data/Base_v1.xlsx`.

---

## Software requirements

The analysis is written in **R**. The main packages include:

```r
tidyverse
readxl
janitor
stringr
psych
EFAtools
GPArotation
cluster
factoextra
NbClust
fpc
seminr
tidymodels
vip
glmnet
ranger
xgboost
mgcv
pROC
ggrepel
pls
leaps
ISLR2
```

You can install missing packages using:

```r
pkgs <- c(
  "tidyverse", "readxl", "janitor", "stringr",
  "psych", "EFAtools", "GPArotation",
  "cluster", "factoextra", "NbClust", "fpc",
  "seminr", "tidymodels", "vip", "glmnet",
  "ranger", "xgboost", "mgcv", "pROC",
  "ggrepel", "pls", "leaps", "ISLR2"
)

to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
```

---

## Quick start

Clone the repository:

```bash
git clone https://github.com/liyoumin/Machine-learning-project.git
cd Machine-learning-project
```

Create data and output folders:

```bash
mkdir -p data outputs
```

Place the raw Excel data file in `data/`, then edit the `DATA_PATH` object in `cluster-prediction.R`.

Run the machine learning pipeline:

```bash
Rscript cluster-prediction.R
```

Run the PLS-SEM pathway model:

```bash
Rscript SEM-PLS.R
```

Compile the manuscript from `main.tex` using Overleaf, TeXShop, VS Code, or another LaTeX workflow.

---

## Expected outputs

Depending on the script section executed, the workflow may generate files such as:

```text
TAB_KMO_Bartlett.txt
TAB_FactorLoadings.csv
TAB_FactorScores.csv
TAB_ClusterSizes.csv
TAB_ClusterProfiles.csv
TAB_NewSegmentProfiles.csv
TAB_CV_Metrics_Classification.csv
FIG_ScreePlot_EFA.pdf
FIG_Silhouette_kmeans.pdf
FIG_Clusters_on_FactorPCs.pdf
FIG_ROC_Combined.pdf
FIG_Importance_RF.pdf
FIG_Importance_XGB.pdf
scored_with_clusters.csv
scored_with_newsegments.csv
```

These outputs support the tables, figures, and model comparisons reported in the manuscript.

---

## Summary of findings

The current preprint reports that consumer perceptions are strong predictors of olive oil choice. Quality perception, taste imagery, price sensitivity, and trust are central drivers of EVOO versus ROO consumption. The machine learning results indicate that relatively interpretable models, especially logistic and LASSO-logit specifications using perception-factor scores, perform competitively against more flexible methods.

In the reported cross-validation results, the LASSO-logit model achieves approximately:

- ROC AUC: **0.91**
- Accuracy: **0.87**
- Log-loss: **0.33**

This suggests that low-dimensional perception constructs provide meaningful predictive power while remaining interpretable for food marketing, consumer behavior, and agricultural economics applications.

---

## Reproducibility notes

Some script sections currently depend on objects created earlier in the workflow, such as `scored_with_clusters`, `dat_model`, `folds`, and saved `.RData` files. For best reproducibility:

1. Run `cluster-prediction.R` before `SEM-PLS.R`.
2. Replace absolute file paths with relative paths.
3. Save all generated tables to an `outputs/` directory.
4. Keep raw data in `data/`, but avoid committing private or restricted survey data.
5. Record package versions using `sessionInfo()` after running the final workflow.

A future improvement would be to convert the workflow into a project-oriented structure using `renv`, `here`, and separate scripts for data cleaning, factor analysis, clustering, modeling, and figure generation.

---

## Suggested citation

If you use this repository or refer to the project, please cite:

```bibtex
@misc{li2026oliveoilml,
  author = {Li, Youmin},
  title = {Consumer Perception Impacts on Olive Oil Choice: A Case Study Using Machine Learning Approach},
  year = {2026},
  note = {Preprint},
  url = {https://liyoumin.github.io/personalweb/publication/preprint/}
}
```

---

## Author

**Youmin Li**  
PhD Student, Food and Resource Economics  
University of Florida

Project webpage: <https://liyoumin.github.io/personalweb/publication/preprint/>

---

## License

No explicit license file is currently provided in this repository. Please contact the author before reusing the code, data, or manuscript materials for publication or commercial purposes.
