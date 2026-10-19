# T2D Microbiome Classification

This repository contains Python scripts for comparing machine-learning models that classify type 2 diabetes (T2D) status from genus-level microbiome features. The main pipelines compare an imbalanced-data baseline with three class-balancing approaches: SMOTE oversampling, ADASYN oversampling, and random undersampling.

## Pipelines

Run a script from the repository root, after preparing its input files:

| Script | Training approach |
| --- | --- |
| `01_baseline_imbalanced.py` | No resampling |
| `02_smote_pipeline.py` | SMOTE oversampling |
| `03_adasyn_pipeline.py` | ADASYN oversampling |
| `04_undersampling_pipeline.py` | Random undersampling |

Each pipeline trains and tunes Random Forest, SVM, Logistic Regression, Linear Discriminant Analysis (LDA), and LightGBM classifiers. It makes a stratified 80/20 train/test split and uses stratified 10-fold cross-validation with ROC AUC for grid-search model selection. In the resampling pipelines, resampling is applied within the training pipeline during cross-validation.

## Input data

The T2D scripts expect these four files in the current working directory:

| File | Expected contents |
| --- | --- |
| `clr_data.csv` | CLR-transformed feature table. The first column contains sample IDs (read as the index); remaining column names are feature/ASV IDs. |
| `metadata.csv` | Sample metadata with `SampleID` and `Group` columns. `Group` must include `T2D`; the scripts treat `T2D` as the positive class. |
| `final.csv` | Taxonomy table with a `Taxon` column containing genus annotations in the form `g__GenusName`, and a feature-ID column matching the IDs in `clr_data.csv`. |
| `final_genus_list.txt` | One genus name per line, matching the names extracted from `Taxon`. |

The scripts match samples between the feature table and metadata, then aggregate matching features by genus. Make sure sample IDs and feature IDs use the same spelling and formatting in their corresponding files. There is a column-name difference in the current scripts: `01_baseline_imbalanced.py`, `03_adasyn_pipeline.py`, and `04_undersampling_pipeline.py` expect the taxonomy feature column to be `Feature.ID`; `02_smote_pipeline.py` expects `Feature ID`. Rename the column in `final.csv` or adjust the relevant script before running if needed.

The input datasets are not included in this repository. Place the files in the project root, or run the scripts from a working directory containing them.

## Setup and run

Use Python with the following packages installed:

```bash
python -m pip install pandas numpy scikit-learn imbalanced-learn lightgbm
```

Then run the desired pipeline from the repository root:

```bash
python 01_baseline_imbalanced.py
python 02_smote_pipeline.py
python 03_adasyn_pipeline.py
python 04_undersampling_pipeline.py
```

The scripts perform multiple grid searches over five model families; expect runs to take substantial time and memory depending on the data and machine. They use fixed random seeds for the train/test split and samplers where applicable.

## Outputs

Each T2D pipeline writes a model-comparison CSV whose name identifies its approach (for example, `SMOTE_Final_Model_Comparison.csv`). Depending on the pipeline, additional CSV files contain ROC AUC and precision-recall results, and some scripts also calculate bootstrap confidence intervals. Output files are written to the current working directory; some supplementary output filenames are shared between scripts, so move or rename results before running another pipeline if you need to retain each run separately.

## Separate benchmark example

`ML(X_Cancer) benchmark.txt` is a separate SVM and LightGBM benchmark example that reads `X_Cancer.csv`. It is not part of the T2D pipelines and is not a Python script with a `.py` extension. Its input data is also not included.

## Scope

These scripts are research/benchmarking code, not a validated diagnostic tool. Model performance depends on the input data, preprocessing, and study design; results should not be interpreted as clinical advice.
